from ultralytics import YOLO
from mmpose.apis import init_model
from mmpose.apis import MMPoseInferencer
# ADD POSE VISUALIZATION HERE:
from mmpose.apis import inference_topdown
import json

from trackers.botsort.bot_sort import BoTSORT
from trackers.multicam_tracker.cluster_track import MCTracker
from trackers.multicam_tracker.clustering import Clustering, ID_Distributor

from perspective_transform.model import PerspectiveTransform
from perspective_transform.calibration import get_calibration_position #calibration_position

from tools.utils_mod import (_COLORS, get_reader_writer, finalize_cams, write_vids, write_results_testset, 
                    update_result_lists_testset, sources, result_paths, cam_ids, cam_ids_full)

# Import 3D bounding box functions
from tools.utils_new import create_3d_bbox, draw_3d_box, project_3d_box_to_bev, visualize

import cv2
import os
import time
import numpy as np
import argparse
import h5py


def make_parser():
    parser = argparse.ArgumentParser(description="Run Online MTPC System")
    parser.add_argument("-s", "--scene", type=str, default=None, help="scene name to inference")
    parser.add_argument("--enable_3d", action="store_true", help="Enable 3D bounding box mode")
    return parser.parse_args()

def write_vids_3d(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, cur_frame, depth_maps=None, calibrations=None, map_img=None, scale_factor=None, tx=None, ty=None, map_h=None):
    outputs = []
    
    # Get feature information from mc_tracker like in original write_vids
    gid_2_lenfeats = {}
    for track in mc_tracker.tracked_mtracks + mc_tracker.lost_mtracks:
        if track.is_activated:
            gid_2_lenfeats[track.track_id] = len(track.features)
        else:
            gid_2_lenfeats[-2] = len(track.features)
    
    for i, (tracker, img, (_, writer), depth_map, calib) in enumerate(zip(trackers, imgs, src_handlers, depth_maps, calibrations)):
        img_2d = img.copy()
        img_3d = img.copy()
        
        # Create outputs for visualization
        outputs_2d = [t.tlbr.tolist() + [t.score, t.global_id, gid_2_lenfeats.get(t.global_id, -1)] for t in tracker.tracked_stracks]
        pose_result = [t.pose for t in tracker.tracked_stracks if t.pose is not None]
        
        # Use original visualization for 2D (with poses)
        img_2d = visualize(outputs_2d, img_2d, _COLORS, pose, pose_result, cur_frame)
        
        # For 3D image, start with pose visualization then add 3D boxes
        img_3d = visualize(outputs_2d, img_3d, _COLORS, pose, pose_result, cur_frame)
        
        # Add 3D bounding boxes on top
        if depth_map is not None and calib:
            K = np.array(calib['intrinsicMatrix'])
            E = np.array(calib['extrinsicMatrix'])
            R = E[:, :3]
            t = E[:, 3].reshape((3, 1))
            
            for track in tracker.tracked_stracks:
                if track.is_activated:
                    bbox = track.tlbr.astype(int)
                    x1, y1, x2, y2 = bbox
                    track_id = track.global_id if hasattr(track, 'global_id') and track.global_id != -1 else track.track_id
                    color_rgb = _COLORS[track_id % len(_COLORS)]
                    color = (int(color_rgb[2] * 255), int(color_rgb[1] * 255), int(color_rgb[0] * 255))
                    
                    try:
                        bbox_2d = [x1, y1, x2, y2]
                        corners_3d_world, center_3d_world, dimensions = create_3d_bbox(
                            bbox_2d, depth_map, K, R, t, depth_scale=1000.0)
                        
                        if corners_3d_world is not None:
                            img_3d = draw_3d_box(img_3d, corners_3d_world, K, R, t, 
                                                color=color, thickness=2)
                            
                            # Add 3D position info
                            wx, wy, wz = center_3d_world
                            pos_label = f"3D: ({wx:.1f}, {wy:.1f}, {wz:.1f})"
                            cv2.putText(img_3d, pos_label, (x1, y2 + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    except Exception as e:
                        print(f"Error creating 3D bbox for track {track_id}: {e}")
        
        # Write frames to respective video writers
        if isinstance(writer, list) and len(writer) >= 2:
            writer[0].write(img_2d)  # 2D video
            writer[1].write(img_3d)  # 3D video
        else:
            if isinstance(writer, list):
                writer[0].write(img_2d)
            else:
                writer.write(img_2d)
        
        outputs.append(outputs_2d)
    
    return outputs

# def write_vids_3d(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, cur_frame, depth_maps=None, calibrations=None, map_img=None, scale_factor=None, tx=None, ty=None, map_h=None):
#     """
#     Modified version of write_vids that supports both 2D and 3D bounding box visualization
#     """
#     outputs = []
    
#     # Get feature information from mc_tracker like in original write_vids
#     gid_2_lenfeats = {}
#     for track in mc_tracker.tracked_mtracks + mc_tracker.lost_mtracks:
#         if track.is_activated:
#             gid_2_lenfeats[track.track_id] = len(track.features)
#         else:
#             gid_2_lenfeats[-2] = len(track.features)
    
#     for i, (tracker, img, (_, writer), depth_map, calib) in enumerate(zip(trackers, imgs, src_handlers, depth_maps, calibrations)):
#         img_2d = img.copy()  # For 2D visualization
#         img_3d = img.copy()  # For 3D visualization
#         map_copy = map_img.copy() if map_img is not None else None
        
#         # Extract camera parameters for 3D processing
#         if calib and depth_map is not None:
#             K = np.array(calib['intrinsicMatrix'])
#             E = np.array(calib['extrinsicMatrix'])
#             R = E[:, :3]
#             t = E[:, 3].reshape((3, 1))
        
#         # Create outputs for original visualization function
#         outputs_2d = [t.tlbr.tolist() + [t.score, t.global_id, gid_2_lenfeats.get(t.global_id, -1)] for t in tracker.tracked_stracks]
        
#         # Use original visualization for 2D
#         # img_2d = visualize(outputs_2d, img_2d, _COLORS, cur_frame)
#         pose_result = [t.pose for t in tracker.tracked_stracks if t.pose is not None]
#         img_2d = visualize(outputs_2d, img_2d, _COLORS, pose, pose_result, cur_frame)
#         img_3d = visualize(outputs_2d, img_3d, _COLORS, pose, pose_result, cur_frame)
        
#         # Draw 3D bounding boxes on img_3d
#         for track in tracker.tracked_stracks:
#             if track.is_activated:
#                 # Get 2D bounding box from tlbr
#                 bbox = track.tlbr.astype(int)
#                 x1, y1, x2, y2 = bbox
                
#                 # Get track color (convert to BGR for OpenCV)
#                 track_id = track.global_id if hasattr(track, 'global_id') and track.global_id != -1 else track.track_id
#                 color_rgb = _COLORS[track_id % len(_COLORS)]
#                 color = (int(color_rgb[2] * 255), int(color_rgb[1] * 255), int(color_rgb[0] * 255))  # Convert RGB to BGR
                
#                 # Draw original 2D box first
#                 cv2.rectangle(img_3d, (x1, y1), (x2, y2), color, 2)
                
#                 # Add track ID and score
#                 len_feats = gid_2_lenfeats.get(track.global_id, -1)
#                 text = f'{track_id} : {track.score * 100:.1f}% | {len_feats}'
#                 cv2.putText(img_3d, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
#                 # Draw 3D bounding box if depth map and calibration are available
#                 if depth_map is not None and calib:
#                     try:
#                         bbox_2d = [x1, y1, x2, y2]
#                         corners_3d_world, center_3d_world, dimensions = create_3d_bbox(
#                             bbox_2d, depth_map, K, R, t, depth_scale=1000.0)
                        
#                         if corners_3d_world is not None and center_3d_world is not None:
#                             # Draw 3D box on the 3D visualization image
#                             img_3d = draw_3d_box(img_3d, corners_3d_world, K, R, t, 
#                                                 color=color, thickness=2)
                            
#                             # Project to bird's eye view if map is available
#                             if map_copy is not None and scale_factor and tx and ty and map_h:
#                                 map_copy = project_3d_box_to_bev(
#                                     corners_3d_world, map_copy, scale_factor, tx, ty, map_h,
#                                     color=color, thickness=2)
                            
#                             # Add 3D position info
#                             wx, wy, wz = center_3d_world
#                             pos_label = f"3D: ({wx:.1f}, {wy:.1f}, {wz:.1f})"
#                             cv2.putText(img_3d, pos_label, (x1, y2 + 20), 
#                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
#                     except Exception as e:
#                         print(f"Error creating 3D bbox for track {track_id}: {e}")
        
#         # Write frames to respective video writers
#         if isinstance(writer, list) and len(writer) >= 2:  # Check if we have multiple writers
#             writer[0].write(img_2d)  # 2D video
#             writer[1].write(img_3d)  # 3D video
#         else:
#             # Fallback to single writer (2D only)
#             if isinstance(writer, list):
#                 writer[0].write(img_2d)
#             else:
#                 writer.write(img_2d)
        
#         outputs.append(outputs_2d)
    
#     return outputs


def get_reader_writer_3d(source_path, enable_3d=False):
    """
    Modified version of get_reader_writer that creates separate writers for 2D and 3D videos
    """
    # Use the same logic as the original get_reader_writer but with multiple writers
    src_paths = sorted(os.listdir(source_path), key=lambda x: int(x.split("_")[-1].split(".")[0].replace("frame", "")))
    src_paths = [os.path.join(source_path, s) for s in src_paths]
    
    fps = 30
    wi, he = 1920, 1080
    
    # Create output directory structure
    scene_name = source_path.split('/')[-2]  # Warehouse_002
    camera_name = source_path.split('/')[-1]  # Camera_0003
    
    os.makedirs(f'output_videos/{scene_name}', exist_ok=True)
    
    writers = []
    
    # Create 2D video writer
    dst_2d = f"output_videos/{scene_name}/{camera_name}_2d.mp4"
    writer_2d = cv2.VideoWriter(dst_2d, cv2.VideoWriter_fourcc(*'mp4v'), fps, (wi, he))
    writers.append(writer_2d)
    
    if enable_3d:
        # Create 3D video writer
        dst_3d = f"output_videos/{scene_name}/{camera_name}_3d.mp4"
        writer_3d = cv2.VideoWriter(dst_3d, cv2.VideoWriter_fourcc(*'mp4v'), fps, (wi, he))
        writers.append(writer_3d)
    
    print(f"{source_path}'s total frames: {len(src_paths)}")
    print(f"Created {'2D and 3D' if enable_3d else '2D only'} video writers")
    
    return [src_paths, writers]


def run(args, conf_thres, iou_thres, sources, result_paths, perspective, cam_ids, scene, enable_3d=False):
    # detection model initilaize
    if int(scene.split('_')[1]) in range(61,81):
        detection = YOLO('runs/best.pt')
    else:
        # detection = YOLO('yolov8n.pt')
        detection = YOLO('runs/best.pt') # Parth' model weights
        print('detection model loaded', detection)
        
    # pose estimation initialize
    #config_file = 'mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-s_8xb256-420e_body8-256x192.py'
    #checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth'
    #pose = init_model(config_file, checkpoint_file, device='cuda:0')
    config_file = 'mmpose/configs/body_2d_keypoint/rtmpose/crowdpose/rtmpose-m_8xb64-210e_crowdpose-256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth'
    pose = init_model(config_file, checkpoint_file, device='cuda:0')
    
    print(f"✅ Pose model loaded successfully with {pose.dataset_meta.get('num_joints', 'unknown')} joints")

    # trackers initialize
    trackers = []
    for i in range(len(sources)):
       trackers.append(BoTSORT(track_buffer=args['track_buffer'], max_batch_size=args['max_batch_size'], 
                            appearance_thresh=args['sct_appearance_thresh'], euc_thresh=args['sct_euclidean_thresh']))

    print("Trackers initialized")
    # perspective transform initialize
    # calibrations = calibration_position[perspective]
    calibrations = get_calibration_position(perspective)
    perspective_transforms = [PerspectiveTransform(c) for c in calibrations]
 
    # id_distributor and multi-camera tracker initialize
    clustering = Clustering(appearance_thresh=args['clt_appearance_thresh'], euc_thresh=args['clt_euclidean_thresh'],
                            match_thresh=0.8)
    mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.8, scene=scene)
    id_distributor = ID_Distributor()

    # get source imgs, video writers - modified for 3D support
    if enable_3d:
        src_handlers = [get_reader_writer_3d(s, enable_3d=True) for s in sources]
    else:
        src_handlers = [get_reader_writer(s) for s in sources]
    
    results_lists = [[] for i in range(len(sources))]  # make empty lists to store tracker outputs in MOT Format

    total_frames = max([len(s[0]) for s in src_handlers])
    cur_frame = 1
    stop = False  
    
    # Load depth data and calibration for 3D processing
    depth_files = []
    camera_calibrations = []
    map_img = None
    scale_factor = None
    tx = None
    ty = None
    map_h = None
    
    if enable_3d:
        # Load depth file
        try:
            depth_file = h5py.File("data/Warehouse_002/depth_maps/Camera_0003.h5", "r")
            print("[Depth] Loaded: Camera_0003.h5")
            depth_files.append(depth_file)
        except Exception as e:
            print(f"[Depth] Failed to load: {e}")
            depth_files.append(None)
        
        # Load calibration data
        calib_path = 'data/Warehouse_002/calibration.json'
        with open(calib_path, 'r') as f:    
            calib_data = json.load(f)
        
        # Get camera calibration for the current scene
        current_cam_id = cam_ids_full[scene]
        cam_num = int(current_cam_id[0].split('_')[1])
        camera_calibration = calib_data['sensors'][cam_num]
        camera_calibrations.append(camera_calibration)
        
        # Load map image for BEV visualization
        try:
            map_path = "data/Warehouse_002/map.png"
            map_img = cv2.imread(map_path)
            if map_img is not None:
                map_h, map_w = map_img.shape[:2]
                scale_factor = camera_calibration.get("scaleFactor", 1.0)
                tx = camera_calibration.get("translationToGlobalCoordinates", {}).get("x", 0)
                ty = camera_calibration.get("translationToGlobalCoordinates", {}).get("y", 0)
                print("[Map] Loaded: map.png")
            else:
                print("[Map] Failed to load map.png")
        except Exception as e:
            print(f"[Map] Error loading map: {e}")
    else:
        depth_files = [None] * len(sources)
        camera_calibrations = [None] * len(sources)

        
    while True:
        # Add frame limit check
        if cur_frame > 1000:  # Process only first 300 frames
            print(f"Reached frame limit of 300, stopping processing...")
            break
        imgs = []
        depth_maps = []
        start = time.time()
        
        # first, run trackers each frame independently
        for idx, ((img_paths, writer), tracker, perspective_transform, result_list) in enumerate(zip(src_handlers, trackers, perspective_transforms, results_lists)):
            # if len(img_paths) == 0 or cur_frame==30:
            if len(img_paths) == 0:
                stop = True
                break
                
            img_path = img_paths.pop(0)
            img_extension = img_path[66:] if len(img_path) > 66 else os.path.basename(img_path)
            img = cv2.imread(img_path)

            # Load depth map if 3D mode is enabled
            depth_map = None
            if enable_3d and depth_files[idx] is not None:
                frame_filename = os.path.splitext(os.path.basename(img_path))[0]
                frame_num_str = frame_filename.replace("frame_", "").zfill(5)
                depth_key = f"distance_to_image_plane_{frame_num_str}.png"
                
                if depth_key in depth_files[idx]:
                    depth_map = depth_files[idx][depth_key][:]
                else:
                    print(f"[Warning] Missing depth map: {depth_key}")
            
            depth_maps.append(depth_map)

            # run detection model
            dets = detection(img, conf=conf_thres, iou=iou_thres, classes=[0,1,2,3], verbose=False)[0].boxes.data.cpu().numpy()

            # Original 3D position calculation (if needed for other parts of the system)
            if enable_3d and depth_map is not None and camera_calibrations[idx] is not None:
                camera_calibration = camera_calibrations[idx]
                
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
            # Add this debug line to confirm pose is being used:
            if cur_frame <= 3:  # Only print for first few frames
                print(f"Frame {cur_frame}: Tracker updated with {len(online_targets)} targets, pose model: {'✅ Used',pose if pose is not None else '❌ None'}")


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
            if enable_3d:
                outputs = write_vids_3d(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, cur_frame,
                                      depth_maps=depth_maps, calibrations=camera_calibrations,
                                      map_img=map_img, scale_factor=scale_factor, tx=tx, ty=ty, map_h=map_h)
            else:
                outputs = write_vids(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, cur_frame)

        print(f"video frame ({cur_frame}/{total_frames})")
        cur_frame += 1
    
    # Clean up
    if enable_3d:
        for (_, writers) in src_handlers:
            for writer in writers:
                writer.release()
                print(f"{writer} released")
    else:
        finalize_cams(src_handlers)
    
    # Close depth files
    if enable_3d:
        for depth_file in depth_files:
            if depth_file is not None:
                depth_file.close()

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

    parsed_args = make_parser()
    scene = parsed_args.scene
    enable_3d = parsed_args.enable_3d

    # scenes = ['scene_000'] # old
    scenes = ['Warehouse_002']
    for scene in scenes:
        run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources[scene], result_paths=result_paths[scene], 
            perspective=scene, cam_ids=cam_ids[scene], scene=scene, enable_3d=enable_3d)

