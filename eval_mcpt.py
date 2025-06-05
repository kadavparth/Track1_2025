from ultralytics import YOLO
from mmpose.apis import init_model
import json

from trackers.botsort.bot_sort import BoTSORT
from trackers.multicam_tracker.cluster_track import MCTracker
from trackers.multicam_tracker.clustering import Clustering, ID_Distributor

from perspective_transform.model import PerspectiveTransform
from perspective_transform.calibration import get_calibration_position #calibration_position

from tools.utils_mod import (_COLORS, get_reader_writer, finalize_cams, write_vids, write_results_testset, 
                    update_result_lists_testset, sources, result_paths, cam_ids, cam_ids_full)

import cv2
import os
import time
import numpy as np
import argparse
import h5py


def make_parser():
    parser = argparse.ArgumentParser(description="Run Online MTPC System")
    parser.add_argument("-s", "--scene", type=str, default=None, help="scene name to inference")
    return parser.parse_args()

def run(args, conf_thres, iou_thres, sources, result_paths, perspective, cam_ids, scene):
    # detection model initilaize
    if int(scene.split('_')[1]) in range(61,81):
        detection = YOLO('pretrained/yolov8x_aic.pt')
    else:
        # detection = YOLO('yolov8n.pt')
        detection = YOLO('runs/best.pt') # Parth' model weights
        print('detection model loaded', detection)

    # pose estimation initialize
    # config_file = '/mmpose/configs/body_2d_keypoint/rtmpose/crowdpose/rtmpose-m_8xb64-210e_crowdpose-256x192.py'
    config_file = 'mmpose/configs/body_2d_keypoint/rtmpose/crowdpose/rtmpose-m_8xb64-210e_crowdpose-256x192.py'
    checkpoint_file = 'pretrained/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth'
    # checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth'
    pose = init_model(config_file, checkpoint_file, device='cuda:0')

    # trackers initialize
    trackers = []
    for i in range(len(sources)):
       trackers.append(BoTSORT(track_buffer=args['track_buffer'], max_batch_size=args['max_batch_size'], 
                            appearance_thresh=args['sct_appearance_thresh'], euc_thresh=args['sct_euclidean_thresh']))

    # perspective transform initialize
    # calibrations = calibration_position[perspective]
    calibrations = get_calibration_position(perspective)
    perspective_transforms = [PerspectiveTransform(c) for c in calibrations]
 
    # id_distributor and multi-camera tracker initialize
    clustering = Clustering(appearance_thresh=args['clt_appearance_thresh'], euc_thresh=args['clt_euclidean_thresh'],
                            match_thresh=0.8)
    mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.8, scene=scene)
    id_distributor = ID_Distributor()

    # get source imgs, video writers
    src_handlers = [get_reader_writer(s) for s in sources]
    results_lists = [[] for i in range(len(sources))]  # make empty lists to store tracker outputs in MOT Format

    total_frames = max([len(s[0]) for s in src_handlers])
    cur_frame = 1
    stop = False  
    
    depth_files = []
    try:
        depth_file = h5py.File("data/Warehouse_002/depth_maps/Camera_0003.h5", "r")
        print("[Depth] Loaded: Camera_0003.h5")
    except Exception as e:
        print(f"[Depth] Failed to load: {e}")

    while True:
        imgs = []
        start = time.time()
        # first, run trackers each frame independently
        for (img_paths, writer), tracker, perspective_transform, result_list in zip(src_handlers, trackers, perspective_transforms, results_lists):
            # if len(img_paths) == 0 or cur_frame==30:
            if len(img_paths) == 0:
                stop = True
                break
            img_path = img_paths.pop(0)
            img_extension = img_path[66:]
            img = cv2.imread(img_path)

            # depth_map = img_extension.replace('.jpg', '.npy')
            # depth_map = np.load(f'data/Warehouse_002/depth_maps/{depth_map}')
            frame_filename = os.path.splitext(os.path.basename(img_path))[0]  # e.g., "frame_00003"
            frame_num_str = frame_filename.replace("frame_", "").zfill(5)
            depth_key = f"distance_to_image_plane_{frame_num_str}.png"
            if depth_file and depth_key in depth_file:
                depth_map = depth_file[depth_key][:]
            else:
                depth_map = None
                print(f"[Warning] Missing depth map: {depth_key}")
            
            # Find the current camera's calibration data
            current_cam_id = cam_ids_full[scene]  # Get current camera ID
            
            # Extract the camera number from the ID (e.g., "Camera_0001" -> 1)
            cam_num = int(current_cam_id[0].split('_')[1])

            # calib_path = f'/workspace/videos/val/{scene}/{current_cam_id[0]}/calibration.json' #old
            calib_path = 'data/Warehouse_002/calibration.json'
            with open(calib_path, 'r') as f:    
                calib_data = json.load(f)

            calib_data = calib_data['sensors']
            camera_calibration = calib_data[cam_num]  # Direct index access

            # run detection model
            dets = detection(img, conf=conf_thres, iou=iou_thres, classes=[0,1,2,3], verbose=False)[0].boxes.data.cpu().numpy()

            dets_cent = [[int((dets[i][0] + dets[i][2])/2), int((dets[i][1] + dets[i][3])/2)] for i in range(len(dets))]
            dets_cent_depth = [depth_map[dets_cent[i][1], dets_cent[i][0]]/1000 for i in range(len(dets_cent))]

            dets_3d = []
            for i in range(len(dets)):
                bbx_cent = [int((dets[i][0] + dets[i][2])/2), int((dets[i][1] + dets[i][3])/2)]
                px_homog = np.array([bbx_cent[0], bbx_cent[1], 1]).reshape(3,1)
                px_world = np.linalg.inv(camera_calibration['intrinsicMatrix']) @ px_homog
                px_world *= dets_cent_depth[i]
                cam_coords_homog = np.append(px_world, 1)
                extrinsic_4x4 = np.eye(4)
                extrinsic_4x4[:3, :4] = camera_calibration['extrinsicMatrix']
                world_coords_homog = np.linalg.inv(extrinsic_4x4) @ cam_coords_homog
                world_coords = world_coords_homog[:3].flatten()
                dets_3d.append(list(world_coords))

            dets_3d = np.array(dets_3d)

            # run tracker
            online_targets, new_ratio = tracker.update(dets, img, img_path, pose)  # run tracker

            # run perspective transform
            perspective_transform.run(tracker, new_ratio)

            # assign temporal global_id to each track for multi-camera tracking
            for t in tracker.tracked_stracks:
                t.t_global_id = id_distributor.assign_id()
            imgs.append(img)
        if stop: break

        # second, run multi-camera tracker using above trackers results
        groups = clustering.update(trackers, cur_frame, scene)
        mc_tracker.update(trackers, groups)
        clustering.update_using_mctracker(trackers, mc_tracker)

        # third, run cluster self-refinements
        if cur_frame % 5 == 0:
            mc_tracker.refinement_clusters()

        # update result lists using updated trackers
        update_result_lists_testset(trackers, results_lists, cur_frame, cam_ids, scene)

        if args['write_vid']:
            outputs = write_vids(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, cur_frame)

        print(f"video frame ({cur_frame}/{total_frames})")
        cur_frame += 1
    
    finalize_cams(src_handlers)

    # save results txt
    write_results_testset(results_lists, result_paths)
    print('Done')


if __name__ == '__main__':
    args = {
        'max_batch_size' : 32,  # maximum input batch size of reid model
        'track_buffer' : 150,  # the frames for keep lost tracks
        'with_reid' : True,  # whether to use reid model's out feature map at first association
        'sct_appearance_thresh' : 0.4,  # threshold of appearance feature cosine distance when do single-cam tracking
        'sct_euclidean_thresh' : 0.1,  # threshold of euclidean distance when do single-cam tracking

        'clt_appearance_thresh' : 0.35,  # threshold of appearance feature cosine distance when do multi-cam clustering
        'clt_euclidean_thresh' : 0.3,  # threshold of euclidean distance when do multi-cam clustering

        'mct_appearance_thresh' : 0.4,  # threshold of appearance feature cosine distance when do cluster tracking (not important)

        'frame_rate' : 30,  # your video(camera)'s fps
        'write_vid' : True,  # write result to video
        }

    scene = make_parser().scene

    # if scene is not None:
    #     run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources[scene], result_paths=result_paths[scene], perspective=scene, cam_ids=cam_ids[scene], scene=scene)
        
    # else:
    #     # run each scene sequentially
        # scenes = [
        #         'scene_061', 'scene_062', 'scene_063', 'scene_064', 'scene_065', 'scene_066', 'scene_067', 'scene_068', 'scene_069', 'scene_070',
        #         'scene_071', 'scene_072', 'scene_073', 'scene_074', 'scene_075', 'scene_076', 'scene_077', 'scene_078', 'scene_079', 'scene_080',
        #         'scene_081', 'scene_082', 'scene_083', 'scene_084', 'scene_085', 'scene_086', 'scene_087', 'scene_088', 'scene_089', 'scene_090',
        #         ]
        # for scene in scenes:
        #     run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources[scene], result_paths=result_paths[scene], perspective=scene, cam_ids=cam_ids[scene], scene=scene)

    # scenes = ['scene_000'] # old
    scenes = ['Warehouse_002']
    for scene in scenes:
        run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources[scene], result_paths=result_paths[scene], perspective=scene, cam_ids=cam_ids[scene], scene=scene)
