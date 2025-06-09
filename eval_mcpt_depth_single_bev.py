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
    parser.add_argument("--enable_bev", action="store_true", help="Enable Bird's Eye View visualization")
    return parser.parse_args()


def world_to_bev_coordinates(world_coords, scale_factor, tx, ty, map_height):
    """
    Convert 3D world coordinates to BEV map pixel coordinates
    
    Args:
        world_coords: [x, y, z] in world coordinate system
        scale_factor: pixels per meter from calibration
        tx, ty: translation to global coordinates
        map_height: height of the map image in pixels
    
    Returns:
        [u, v] pixel coordinates in BEV map
    """
    world_x, world_y = world_coords[0], world_coords[1]
    
    # Convert world coordinates to map coordinates
    map_x = (world_x + tx) * scale_factor
    map_y = (world_y + ty) * scale_factor
    
    # Convert to image coordinates (flip Y axis)
    pixel_u = int(map_x)
    pixel_v = int(map_height - map_y)  # Flip Y coordinate
    
    return [pixel_u, pixel_v]


def draw_3d_box_on_bev(bev_img, corners_3d_world, scale_factor, tx, ty, map_height, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding box projected onto BEV map
    
    Args:
        bev_img: BEV map image
        corners_3d_world: 8 corners of the 3D box in world coordinates
        scale_factor, tx, ty: calibration parameters
        map_height: height of the BEV map
        color: drawing color
        thickness: line thickness
    
    Returns:
        BEV image with drawn box
    """
    if corners_3d_world is None:
        return bev_img
    
    # Convert all corners to BEV coordinates
    bev_corners = []
    for corner in corners_3d_world:
        bev_coord = world_to_bev_coordinates(corner, scale_factor, tx, ty, map_height)
        bev_corners.append(bev_coord)
    
    bev_corners = np.array(bev_corners, dtype=np.int32)
    
    # Draw the bottom face of the box (feet level)
    bottom_face = bev_corners[:4]  # First 4 corners are bottom face
    cv2.polylines(bev_img, [bottom_face], isClosed=True, color=color, thickness=thickness)
    
    # Draw the top face of the box (head level)
    top_face = bev_corners[4:]  # Last 4 corners are top face
    cv2.polylines(bev_img, [top_face], isClosed=True, color=color, thickness=thickness)
    
    # Draw vertical lines connecting bottom and top
    for i in range(4):
        pt1 = tuple(bev_corners[i])
        pt2 = tuple(bev_corners[i + 4])
        cv2.line(bev_img, pt1, pt2, color, thickness)
    
    # Draw center point
    center_3d = np.mean(corners_3d_world, axis=0)
    center_bev = world_to_bev_coordinates(center_3d, scale_factor, tx, ty, map_height)
    cv2.circle(bev_img, tuple(center_bev), 4, color, -1)
    
    return bev_img


def get_reader_writer_with_bev(source_path, enable_3d=False, enable_bev=False):
    """
    Modified version that creates BEV writer if needed
    """
    # Use the same logic as the original get_reader_writer but with BEV writer
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
    
    if enable_bev:
        # Create BEV video writer (will be set up later when we know map dimensions)
        writers.append(None)  # Placeholder for BEV writer
    
    print(f"{source_path}'s total frames: {len(src_paths)}")
    print(f"Created {'2D' + (', 3D' if enable_3d else '') + (', BEV' if enable_bev else '')} video writers")
    
    return [src_paths, writers]


def write_vids_with_bev(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, cur_frame, 
                       depth_maps=None, calibrations=None, map_img=None, bev_writer=None,
                       scale_factor=None, tx=None, ty=None, map_h=None, enable_3d=False, enable_bev=False):
    """
    Enhanced version that includes BEV visualization
    """
    outputs = []
    
    # Get feature information from mc_tracker like in original write_vids
    gid_2_lenfeats = {}
    for track in mc_tracker.tracked_mtracks + mc_tracker.lost_mtracks:
        if track.is_activated:
            gid_2_lenfeats[track.track_id] = len(track.features)
        else:
            gid_2_lenfeats[-2] = len(track.features)
    
    # Create BEV frame if enabled
    bev_frame = None
    if enable_bev and map_img is not None:
        bev_frame = map_img.copy()
    
    for i, (tracker, img, (_, writers), depth_map, calib) in enumerate(zip(trackers, imgs, src_handlers, depth_maps, calibrations)):
        img_2d = img.copy()
        img_3d = img.copy() if enable_3d else None
        
        # Create outputs for visualization
        outputs_2d = [t.tlbr.tolist() + [t.score, t.global_id, gid_2_lenfeats.get(t.global_id, -1)] for t in tracker.tracked_stracks]
        pose_result = [t.pose for t in tracker.tracked_stracks if t.pose is not None]
        
        # Use original visualization for 2D (with poses)
        img_2d = visualize(outputs_2d, img_2d, _COLORS, pose, pose_result, cur_frame)
        
        # For 3D image, start with pose visualization then add 3D boxes
        if enable_3d and img_3d is not None:
            img_3d = visualize(outputs_2d, img_3d, _COLORS, pose, pose_result, cur_frame)
        
        # Process 3D bounding boxes and BEV visualization
        if depth_map is not None and calib and (enable_3d or enable_bev):
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
                            # Draw 3D box on camera view
                            if enable_3d and img_3d is not None:
                                img_3d = draw_3d_box(img_3d, corners_3d_world, K, R, t, 
                                                    color=color, thickness=2)
                                
                                # Add 3D position info
                                wx, wy, wz = center_3d_world
                                pos_label = f"3D: ({wx:.1f}, {wy:.1f}, {wz:.1f})"
                                cv2.putText(img_3d, pos_label, (x1, y2 + 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
                            # Draw on BEV map
                            if enable_bev and bev_frame is not None and scale_factor and tx and ty and map_h:
                                bev_frame = draw_3d_box_on_bev(
                                    bev_frame, corners_3d_world, scale_factor, tx, ty, map_h,
                                    color=color, thickness=2)
                                
                                # Add track ID label on BEV
                                center_bev = world_to_bev_coordinates(center_3d_world, scale_factor, tx, ty, map_h)
                                cv2.putText(bev_frame, f'{track_id}', 
                                          (center_bev[0] + 10, center_bev[1] - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
                    except Exception as e:
                        print(f"Error creating 3D bbox for track {track_id}: {e}")
        
        # Write frames to respective video writers
        writer_idx = 0
        
        # 2D video
        if len(writers) > writer_idx and writers[writer_idx] is not None:
            writers[writer_idx].write(img_2d)
        writer_idx += 1
        
        # 3D video
        if enable_3d and len(writers) > writer_idx and writers[writer_idx] is not None:
            writers[writer_idx].write(img_3d)
        if enable_3d:
            writer_idx += 1
        
        outputs.append(outputs_2d)
    
    # Write BEV frame
    if enable_bev and bev_writer is not None and bev_frame is not None:
        bev_writer.write(bev_frame)
    
    return outputs


def run(args, conf_thres, iou_thres, sources, result_paths, perspective, cam_ids, scene, enable_3d=False, enable_bev=False):
    # detection model initilaize
    if int(scene.split('_')[1]) in range(61,81):
        detection = YOLO('runs/best.pt')
    else:
        # detection = YOLO('yolov8n.pt')
        detection = YOLO('runs/best.pt') # Parth' model weights
        print('detection model loaded', detection)
        
    # pose estimation initialize
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
    calibrations = get_calibration_position(perspective)
    perspective_transforms = [PerspectiveTransform(c) for c in calibrations]
 
    # id_distributor and multi-camera tracker initialize
    clustering = Clustering(appearance_thresh=args['clt_appearance_thresh'], euc_thresh=args['clt_euclidean_thresh'],
                            match_thresh=0.8)
    mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.8, scene=scene)
    id_distributor = ID_Distributor()

    # get source imgs, video writers - modified for 3D and BEV support
    src_handlers = [get_reader_writer_with_bev(s, enable_3d=enable_3d, enable_bev=enable_bev) for s in sources]
    
    results_lists = [[] for i in range(len(sources))]  # make empty lists to store tracker outputs in MOT Format

    total_frames = max([len(s[0]) for s in src_handlers])
    cur_frame = 1
    stop = False  
    
    # Load depth data, calibration, and BEV setup
    depth_files = []
    camera_calibrations = []
    map_img = None
    bev_writer = None
    scale_factor = None
    tx = None
    ty = None
    map_h = None
    
    if enable_3d or enable_bev:
        # Load depth file
        try:
            depth_file = h5py.File("data/Warehouse_002/depth_maps/Camera_0003.h5", "r")
            print("[Depth] Loaded: Camera_0003.h5")
            depth_files.append(depth_file)
        except Exception as e:
            print(f"[Depth] Failed to load: {e}")
            depth_files.append(None)
        
        # Load calibration data from the provided JSON
        calib_path = 'data/Warehouse_002/calibration.json'
        with open(calib_path, 'r') as f:    
            calib_data = json.load(f)
        
        # Get camera calibration for the current scene
        current_cam_id = cam_ids_full[scene]
        cam_num = int(current_cam_id[0].split('_')[1])
        camera_calibration = calib_data['sensors'][cam_num]
        camera_calibrations.append(camera_calibration)
        
        # Setup BEV visualization if enabled
        if enable_bev:
            try:
                map_path = "data/Warehouse_002/map.png"
                map_img = cv2.imread(map_path)
                if map_img is not None:
                    map_h, map_w = map_img.shape[:2]
                    
                    # Get BEV parameters from calibration
                    scale_factor = camera_calibration.get("scaleFactor", 14.420706776686613)
                    tx = camera_calibration.get("translationToGlobalCoordinates", {}).get("x", 76.54132562910155)
                    ty = camera_calibration.get("translationToGlobalCoordinates", {}).get("y", 41.46934389588847)
                    
                    print(f"[BEV] Map loaded: {map_w}x{map_h}, scale: {scale_factor}, translation: ({tx}, {ty})")
                    
                    # Create BEV video writer
                    scene_name = sources[0].split('/')[-2]
                    bev_output_path = f"output_videos/{scene_name}/bev_tracking.mp4"
                    fps = 30
                    bev_writer = cv2.VideoWriter(bev_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (map_w, map_h))
                    
                    # Update the placeholder in src_handlers
                    for handler in src_handlers:
                        if len(handler[1]) > 2 and handler[1][2] is None:
                            handler[1][2] = bev_writer
                    
                    print(f"[BEV] Video writer created: {bev_output_path}")
                else:
                    print("[BEV] Failed to load map.png")
                    enable_bev = False
            except Exception as e:
                print(f"[BEV] Error loading map: {e}")
                enable_bev = False
    else:
        depth_files = [None] * len(sources)
        camera_calibrations = [None] * len(sources)

        
    while True:
        # Add frame limit check
        if cur_frame > 1000:  # Process only first 1000 frames
            print(f"Reached frame limit of 1000, stopping processing...")
            break
        imgs = []
        depth_maps = []
        start = time.time()
        
        # first, run trackers each frame independently
        for idx, ((img_paths, writers), tracker, perspective_transform, result_list) in enumerate(zip(src_handlers, trackers, perspective_transforms, results_lists)):
            if len(img_paths) == 0:
                stop = True
                break
                
            img_path = img_paths.pop(0)
            img_extension = img_path[66:] if len(img_path) > 66 else os.path.basename(img_path)
            img = cv2.imread(img_path)

            # Load depth map if 3D mode is enabled
            depth_map = None
            if (enable_3d or enable_bev) and depth_files[idx] is not None:
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
            if (enable_3d or enable_bev) and depth_map is not None and camera_calibrations[idx] is not None:
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
                print(f"Frame {cur_frame}: Tracker updated with {len(online_targets)} targets, pose model: {'✅ Used' if pose is not None else '❌ None'}")

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
            outputs = write_vids_with_bev(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, cur_frame,
                                        depth_maps=depth_maps, calibrations=camera_calibrations,
                                        map_img=map_img, bev_writer=bev_writer,
                                        scale_factor=scale_factor, tx=tx, ty=ty, map_h=map_h,
                                        enable_3d=enable_3d, enable_bev=enable_bev)

        print(f"video frame ({cur_frame}/{total_frames})")
        cur_frame += 1
    
    # Clean up
    for (_, writers) in src_handlers:
        for writer in writers:
            if writer is not None:
                writer.release()
    
    if bev_writer is not None:
        bev_writer.release()
        print("BEV video writer released")
    
    # Close depth files
    if enable_3d or enable_bev:
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
    enable_bev = parsed_args.enable_bev

    # scenes = ['scene_000'] # old
    scenes = ['Warehouse_002']
    for scene in scenes:
        run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources[scene], result_paths=result_paths[scene], 
            perspective=scene, cam_ids=cam_ids[scene], scene=scene, enable_3d=enable_3d, enable_bev=enable_bev)