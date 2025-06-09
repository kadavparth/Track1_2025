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

from tools.utils_mod import (_COLORS, finalize_cams, write_vids, write_results_testset, 
                    update_result_lists_testset, sources, result_paths, cam_ids, cam_ids_full)

# Import 3D bounding box functions
from tools.utils_new import create_3d_bbox, draw_3d_box, project_3d_box_to_bev, visualize

import cv2
import os
import time
import numpy as np
import argparse
import h5py
import glob
try:
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[Warning] scipy not available - using fallback for BEV box drawing")


def make_parser():
    parser = argparse.ArgumentParser(description="Run Online MTPC System")
    parser.add_argument("-s", "--scene", type=str, default="Warehouse_002", help="scene name to inference")
    parser.add_argument("--enable_3d", action="store_true", help="Enable 3D bounding box mode")
    parser.add_argument("--data_root", type=str, default="data/Warehouse_002", help="Root directory containing videos and depth maps")
    parser.add_argument("--save_logs", action="store_true", help="Save terminal output to log file")
    parser.add_argument("--log_file", type=str, default="tracking_log.txt", help="Log file path")
    parser.add_argument("--bev_scale_multiplier", type=float, default=1.0, help="Scale multiplier for BEV 3D boxes (default: 1.0)")
    parser.add_argument("--debug_bev", action="store_true", help="Enable BEV debugging output")
    return parser.parse_args()


def get_video_readers_and_writers(data_root, scene_name, enable_3d=False, enable_bev=False):
    """
    Get video readers from MP4 files and create corresponding writers
    
    Args:
        data_root: Root directory containing videos and depth maps
        scene_name: Name of the scene (e.g., "Warehouse_002")
        enable_3d: Whether to create 3D visualization writers
        enable_bev: Whether to create BEV visualization writer
    
    Returns:
        List of [video_reader, writers] for each camera
    """
    video_pattern = os.path.join(data_root, "*.mp4")
    video_files = sorted(glob.glob(video_pattern))
    
    if not video_files:
        raise ValueError(f"No MP4 files found in {data_root}")
    
    print(f"Found {len(video_files)} camera videos:")
    for video_file in video_files:
        print(f"  - {os.path.basename(video_file)}")
    
    src_handlers = []
    
    for video_file in video_files:
        camera_name = os.path.splitext(os.path.basename(video_file))[0]  # e.g., "Camera_0001"
        
        # Create video reader
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"{camera_name}: {frame_count} frames, {fps} FPS, {width}x{height}")
        
        # Create output directory
        output_dir = f'output_videos/{scene_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create video writers
        writers = []
        
        # 2D video writer
        dst_2d = f"{output_dir}/{camera_name}_2d.mp4"
        writer_2d = cv2.VideoWriter(dst_2d, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        writers.append(writer_2d)
        
        if enable_3d:
            # 3D video writer
            dst_3d = f"{output_dir}/{camera_name}_3d.mp4"
            writer_3d = cv2.VideoWriter(dst_3d, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            writers.append(writer_3d)
        
        src_handlers.append([cap, writers, camera_name, frame_count])
    
    # Add BEV video writer (shared across all cameras)
    if enable_bev:
        bev_dst = f"{output_dir}/bev_view.mp4"
        # We'll determine BEV dimensions when we load the map
        src_handlers.append(['bev_writer_placeholder', bev_dst])
    
    return src_handlers


def load_depth_maps(data_root, camera_names):
    """
    Load depth map files for each camera
    
    Args:
        data_root: Root directory containing depth maps
        camera_names: List of camera names (e.g., ["Camera_0001", "Camera_0002"])
    
    Returns:
        List of depth file handles or None if not found
    """
    depth_files = []
    depth_dir = os.path.join(data_root, "depth_maps")
    
    for camera_name in camera_names:
        depth_file_path = os.path.join(depth_dir, f"{camera_name}.h5")
        
        try:
            depth_file = h5py.File(depth_file_path, "r")
            print(f"[Depth] Loaded: {camera_name}.h5")
            depth_files.append(depth_file)
        except Exception as e:
            print(f"[Depth] Failed to load {camera_name}.h5: {e}")
            depth_files.append(None)
    
    return depth_files


def load_calibration_data(data_root, camera_names):
    """
    Load camera calibration data for each camera with proper ID mapping
    
    Args:
        data_root: Root directory containing calibration.json
        camera_names: List of camera names (e.g., ["Camera_0001", "Camera_0003"])
    
    Returns:
        List of calibration dictionaries
    """
    calib_path = os.path.join(data_root, 'calibration.json')
    
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
    
    camera_calibrations = []
    
    # Create mapping from camera names to sensor indices
    camera_id_mapping = {}
    for i, sensor in enumerate(calib_data['sensors']):
        sensor_id = sensor.get('id', '')
        if sensor_id == 'Camera':
            camera_id_mapping['Camera_0000'] = i  # Default camera
        else:
            # Convert Camera_01 -> Camera_0001, Camera_03 -> Camera_0003, etc.
            if '_' in sensor_id:
                cam_num = sensor_id.split('_')[1]
                formatted_name = f"Camera_{cam_num.zfill(4)}"
                camera_id_mapping[formatted_name] = i
    
    print(f"[Calibration] Camera ID mapping: {camera_id_mapping}")
    
    for camera_name in camera_names:
        if camera_name in camera_id_mapping:
            sensor_idx = camera_id_mapping[camera_name]
            camera_calibration = calib_data['sensors'][sensor_idx]
            camera_calibrations.append(camera_calibration)
            print(f"[Calibration] Loaded for {camera_name} (sensor index {sensor_idx})")
        else:
            print(f"[Calibration] Not found for {camera_name}")
            camera_calibrations.append(None)
    
    return camera_calibrations


def load_bev_map_and_params(data_root, camera_calibrations):
    """
    Load the BEV map image and extract global coordinate transformation parameters
    
    Args:
        data_root: Root directory containing map.png
        camera_calibrations: List of camera calibration data
    
    Returns:
        Tuple of (map_img, scale_factor, tx, ty, map_h, map_w, bev_writer)
    """
    # Load map image
    map_path = os.path.join(data_root, "map.png")
    map_img = None
    scale_factor = None
    tx = None
    ty = None
    map_h = None
    map_w = None
    
    try:
        map_img = cv2.imread(map_path)
        if map_img is not None:
            map_h, map_w = map_img.shape[:2]
            
            # Get global coordinate parameters from any valid calibration
            # (they should be the same for all cameras in the same scene)
            for calib in camera_calibrations:
                if calib is not None:
                    scale_factor = calib.get("scaleFactor", 14.420706776686613)
                    translation = calib.get("translationToGlobalCoordinates", {})
                    tx = translation.get("x", 76.54132562910155)
                    ty = translation.get("y", 41.46934389588847)
                    break
            
            print(f"[BEV] Loaded map: {map_path} ({map_w}x{map_h})")
            print(f"[BEV] Scale factor: {scale_factor}")
            print(f"[BEV] Translation: ({tx}, {ty})")
        else:
            print(f"[BEV] Failed to load map: {map_path}")
    except Exception as e:
        print(f"[BEV] Error loading map: {e}")
    
    return map_img, scale_factor, tx, ty, map_h, map_w


def create_bev_writers(output_dir, scene_name, map_w, map_h, fps=30):
    """
    Create BEV video writers with map dimensions - both dots and 3D boxes versions
    """
    # BEV with dots
    bev_dots_dst = f"{output_dir}/{scene_name}_bev_dots.mp4"
    bev_dots_writer = cv2.VideoWriter(bev_dots_dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (map_w, map_h))
    
    # BEV with 3D bounding boxes
    bev_boxes_dst = f"{output_dir}/{scene_name}_bev_boxes.mp4"
    bev_boxes_writer = cv2.VideoWriter(bev_boxes_dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (map_w, map_h))
    
    print(f"[BEV] Created video writers:")
    print(f"  - Dots: {bev_dots_dst} ({map_w}x{map_h})")
    print(f"  - Boxes: {bev_boxes_dst} ({map_w}x{map_h})")
    
    return bev_dots_writer, bev_boxes_writer


def project_corners_to_bev(corners_3d_world, scale_factor, tx, ty, map_h):
    """
    Project 3D bounding box corners to BEV map coordinates
    
    Args:
        corners_3d_world: 8x3 array of 3D corner coordinates in world space
        scale_factor: Scale factor from calibration
        tx, ty: Translation offsets from calibration  
        map_h: Height of the BEV map
        
    Returns:
        List of (x, y) coordinates in BEV map space
    """
    bev_corners = []
    for corner in corners_3d_world:
        wx, wy, wz = corner
        # Convert world coordinates to BEV map coordinates
        map_x = int((wx + tx) * scale_factor)
        map_y = int(map_h - (wy + ty) * scale_factor)
        bev_corners.append((map_x, map_y))
    return bev_corners


def draw_3d_box_on_bev(bev_map, corners_3d_world, scale_factor, tx, ty, map_h, color, thickness=2, track_id=None, debug=False):
    """
    Draw a 3D bounding box on the BEV map with proper scaling
    
    Args:
        bev_map: BEV map image
        corners_3d_world: 8x3 array of 3D corner coordinates (in meters)
        scale_factor: Scale factor from calibration (pixels per meter) - ~14.42
        tx, ty: Translation offsets from calibration (meters)
        map_h: Height of the BEV map
        color: BGR color tuple
        thickness: Line thickness
        track_id: Optional track ID to display
        debug: Whether to print debug information
    
    Returns:
        Updated BEV map
    """
    if corners_3d_world is None:
        return bev_map
    
    # Debug: Print world coordinates and dimensions
    if debug and track_id is not None:
        min_coords = np.min(corners_3d_world, axis=0)
        max_coords = np.max(corners_3d_world, axis=0)
        dimensions = max_coords - min_coords
        print(f"[DEBUG] Track {track_id}: World dimensions = {dimensions[0]:.2f}m x {dimensions[1]:.2f}m x {dimensions[2]:.2f}m")
        print(f"[DEBUG] Track {track_id}: Center = ({np.mean(corners_3d_world[:, 0]):.2f}, {np.mean(corners_3d_world[:, 1]):.2f}, {np.mean(corners_3d_world[:, 2]):.2f})")
        
    # Project 3D corners to BEV coordinates
    bev_corners = project_corners_to_bev(corners_3d_world, scale_factor, tx, ty, map_h)
    
    # Filter corners that are within map bounds
    valid_corners = [(x, y) for x, y in bev_corners 
                    if 0 <= x < bev_map.shape[1] and 0 <= y < bev_map.shape[0]]
    
    if len(valid_corners) < 3:
        if debug:
            print(f"[DEBUG] Track {track_id}: Only {len(valid_corners)} valid corners out of {len(bev_corners)}")
        return bev_map
    
    # Convert to numpy array for easier handling
    bev_corners_np = np.array(valid_corners)
    
    # For BEV, we want to show the footprint (bottom face) of the 3D box
    # The corners are typically ordered as: bottom 4 corners [0-3], then top 4 corners [4-7]
    if len(bev_corners) >= 8:
        # Use bottom 4 corners for the footprint
        bottom_indices = [0, 1, 2, 3]  # Assuming standard corner ordering
        bottom_corners_3d = corners_3d_world[bottom_indices]
        bottom_corners_bev = [bev_corners[i] for i in bottom_indices if 0 <= bev_corners[i][0] < bev_map.shape[1] and 0 <= bev_corners[i][1] < bev_map.shape[0]]
        
        if len(bottom_corners_bev) >= 3:
            bottom_corners_np = np.array(bottom_corners_bev)
            
            if debug and track_id is not None:
                print(f"[DEBUG] Track {track_id}: BEV corners = {bottom_corners_np}")
                bev_width = np.max(bottom_corners_np[:, 0]) - np.min(bottom_corners_np[:, 0])
                bev_height = np.max(bottom_corners_np[:, 1]) - np.min(bottom_corners_np[:, 1])
                print(f"[DEBUG] Track {track_id}: BEV size = {bev_width} x {bev_height} pixels")
                print(f"[DEBUG] Track {track_id}: Real size = {bev_width/scale_factor:.2f} x {bev_height/scale_factor:.2f} meters")
        else:
            bottom_corners_np = bev_corners_np[:4] if len(bev_corners_np) >= 4 else bev_corners_np
    else:
        # Use all available corners
        bottom_corners_np = bev_corners_np
    
    # Draw the footprint
    if len(bottom_corners_np) >= 3:
        # Create a convex hull to get the proper order
        try:
            if SCIPY_AVAILABLE and len(bottom_corners_np) >= 3:
                hull = ConvexHull(bottom_corners_np)
                hull_corners = bottom_corners_np[hull.vertices]
                
                # Draw the polygon outline
                cv2.polylines(bev_map, [hull_corners.astype(np.int32)], True, color, thickness)
                
                # Fill the polygon with semi-transparent color
                overlay = bev_map.copy()
                cv2.fillPoly(overlay, [hull_corners.astype(np.int32)], color)
                cv2.addWeighted(bev_map, 0.7, overlay, 0.3, 0, bev_map)
            else:
                # Fallback: draw lines between consecutive corners
                for i in range(len(bottom_corners_np)):
                    start_point = tuple(bottom_corners_np[i].astype(int))
                    end_point = tuple(bottom_corners_np[(i + 1) % len(bottom_corners_np)].astype(int))
                    cv2.line(bev_map, start_point, end_point, color, thickness)
                    
                # Draw a filled circle at the center as fallback
                center_x = int(np.mean(bottom_corners_np[:, 0]))
                center_y = int(np.mean(bottom_corners_np[:, 1]))
                cv2.circle(bev_map, (center_x, center_y), max(2, thickness), color, -1)
        except Exception as e:
            if debug:
                print(f"[DEBUG] Track {track_id}: Error drawing polygon: {e}")
            # Fallback: draw lines between consecutive corners
            for i in range(len(bottom_corners_np)):
                start_point = tuple(bottom_corners_np[i].astype(int))
                end_point = tuple(bottom_corners_np[(i + 1) % len(bottom_corners_np)].astype(int))
                cv2.line(bev_map, start_point, end_point, color, thickness)
    
    # Add track ID at the center
    if track_id is not None and len(valid_corners) > 0:
        center_x = int(np.mean([x for x, y in valid_corners]))
        center_y = int(np.mean([y for x, y in valid_corners]))
        cv2.putText(bev_map, f'{track_id}', (center_x-10, center_y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return bev_map


def write_vids_3d_with_bev(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, cur_frame, 
                          depth_maps=None, calibrations=None, map_img=None, scale_factor=None, 
                          tx=None, ty=None, map_h=None, bev_dots_writer=None, bev_boxes_writer=None,
                          bev_scale_multiplier=1.0, debug_bev=False):
    """
    Modified version of write_vids that supports 2D, 3D, and dual BEV visualization
    """
    outputs = []
    
    # Get feature information from mc_tracker
    gid_2_lenfeats = {}
    for track in mc_tracker.tracked_mtracks + mc_tracker.lost_mtracks:
        if track.is_activated:
            gid_2_lenfeats[track.track_id] = len(track.features)
        else:
            gid_2_lenfeats[-2] = len(track.features)
    
    # Initialize BEV maps for this frame
    bev_dots_map = None
    bev_boxes_map = None
    if map_img is not None:
        bev_dots_map = map_img.copy()
        bev_boxes_map = map_img.copy()
    
    for i, (tracker, img, (_, writers, camera_name, _), depth_map, calib) in enumerate(zip(trackers, imgs, src_handlers, depth_maps, calibrations)):
        img_2d = img.copy()
        img_3d = img.copy()
        
        # Create outputs for visualization
        outputs_2d = [t.tlbr.tolist() + [t.score, t.global_id, gid_2_lenfeats.get(t.global_id, -1)] for t in tracker.tracked_stracks]
        pose_result = [t.pose for t in tracker.tracked_stracks if t.pose is not None]
        
        # Use original visualization for 2D (with poses)
        img_2d = visualize(outputs_2d, img_2d, _COLORS, pose, pose_result, cur_frame)
        
        # For 3D image, start with pose visualization then add 3D boxes
        img_3d = visualize(outputs_2d, img_3d, _COLORS, pose, pose_result, cur_frame)
        
        # Add 3D bounding boxes and project to BEV
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
                            # Draw 3D box on camera view
                            img_3d = draw_3d_box(img_3d, corners_3d_world, K, R, t, 
                                                color=color, thickness=2)
                            
                            # Project to both BEV views if map is available
                            if (bev_dots_map is not None and bev_boxes_map is not None and 
                                scale_factor and tx is not None and ty is not None and map_h):
                                
                                # BEV with dots (original approach)
                                if center_3d_world is not None:
                                    wx, wy, wz = center_3d_world
                                    # Convert world coordinates to BEV map coordinates
                                    map_x = int((wx + tx) * scale_factor)
                                    map_y = int(map_h - (wy + ty) * scale_factor)
                                    
                                    # Ensure coordinates are within map bounds
                                    if 0 <= map_x < bev_dots_map.shape[1] and 0 <= map_y < bev_dots_map.shape[0]:
                                        # Draw a circle for the person's position
                                        cv2.circle(bev_dots_map, (map_x, map_y), 8, color, -1)
                                        cv2.putText(bev_dots_map, f'{track_id}', (map_x+12, map_y+5), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                                # BEV with 3D bounding boxes (enhanced approach)
                                adjusted_scale_factor = scale_factor * bev_scale_multiplier
                                
                                if debug_bev and cur_frame <= 5:  # Debug first few frames
                                    print(f"[BEV DEBUG] Frame {cur_frame}, Track {track_id}:")
                                    print(f"  Original scale factor: {scale_factor}")
                                    print(f"  Scale multiplier: {bev_scale_multiplier}")
                                    print(f"  Adjusted scale factor: {adjusted_scale_factor}")
                                    print(f"  Translation: ({tx}, {ty})")
                                    print(f"  Map height: {map_h}")
                                
                                bev_boxes_map = draw_3d_box_on_bev(
                                    bev_boxes_map, corners_3d_world, adjusted_scale_factor, tx, ty, map_h,
                                    color=color, thickness=3, track_id=track_id, debug=debug_bev and cur_frame <= 5)
                            
                            # Add 3D position info to camera view
                            wx, wy, wz = center_3d_world
                            pos_label = f"3D: ({wx:.1f}, {wy:.1f}, {wz:.1f})"
                            cv2.putText(img_3d, pos_label, (x1, y2 + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    except Exception as e:
                        print(f"Error creating 3D bbox for track {track_id} in {camera_name}: {e}")
        
        # Write frames to respective video writers
        if len(writers) >= 2:
            writers[0].write(img_2d)  # 2D video
            writers[1].write(img_3d)  # 3D video
        else:
            writers[0].write(img_2d)  # 2D only
        
        outputs.append(outputs_2d)
    
    # Write BEV frames
    if bev_dots_writer is not None and bev_dots_map is not None:
        # Add frame number and timestamp to BEV dots
        cv2.putText(bev_dots_map, f'Frame: {cur_frame} (Dots)', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        bev_dots_writer.write(bev_dots_map)
    
    if bev_boxes_writer is not None and bev_boxes_map is not None:
        # Add frame number and timestamp to BEV boxes
        cv2.putText(bev_boxes_map, f'Frame: {cur_frame} (3D Boxes)', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        bev_boxes_writer.write(bev_boxes_map)
    
    return outputs


def finalize_cameras_with_bev(src_handlers, bev_dots_writer=None, bev_boxes_writer=None):
    """
    Release video readers and writers including dual BEV writers
    """
    for cap, writers, camera_name, _ in src_handlers:
        cap.release()
        for writer in writers:
            writer.release()
        print(f"Released {camera_name}")
    
    if bev_dots_writer is not None:
        bev_dots_writer.release()
        print("Released BEV dots writer")
    
    if bev_boxes_writer is not None:
        bev_boxes_writer.release()
        print("Released BEV boxes writer")


def run(args, conf_thres, iou_thres, data_root, scene, enable_3d=False, log_file=None, bev_scale_multiplier=1.0, debug_bev=False):
    # Setup logging if requested
    if log_file:
        import sys
        class Logger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, "w")
            
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()
            
            def flush(self):
                self.terminal.flush()
                self.log.flush()
        
        sys.stdout = Logger(log_file)
        print(f"[LOG] Starting logging to {log_file}")
    
    print(f"[SYSTEM] Starting multi-camera tracking for scene: {scene}")
    print(f"[SYSTEM] Data root: {data_root}")
    print(f"[SYSTEM] 3D mode: {'Enabled' if enable_3d else 'Disabled'}")
    
    # Detection model initialize
    detection = YOLO('runs/best.pt')
    print('[DETECTION] Model loaded:', detection)
        
    # Pose estimation initialize
    config_file = 'mmpose/configs/body_2d_keypoint/rtmpose/crowdpose/rtmpose-m_8xb64-210e_crowdpose-256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth'
    pose = init_model(config_file, checkpoint_file, device='cuda:0')
    print(f"[POSE] Model loaded successfully with {pose.dataset_meta.get('num_joints', 'unknown')} joints")

    # Get video readers and writers for all cameras
    enable_bev = enable_3d  # Enable BEV when 3D is enabled
    src_handlers = get_video_readers_and_writers(data_root, scene, enable_3d, enable_bev)
    camera_names = [handler[2] for handler in src_handlers if len(handler) == 4]  # Filter out BEV placeholder
    print(f"[CAMERAS] Found cameras: {camera_names}")
    
    # Initialize trackers for each camera
    trackers = []
    for i in range(len(camera_names)):
       trackers.append(BoTSORT(track_buffer=args['track_buffer'], max_batch_size=args['max_batch_size'], 
                            appearance_thresh=args['sct_appearance_thresh'], euc_thresh=args['sct_euclidean_thresh']))
    print(f"[TRACKERS] Initialized for {len(trackers)} cameras")

    # Create camera ID mapping for perspective transforms
    camera_id_mapping = {}
    for camera_name in camera_names:
        if camera_name == 'Camera_0000':
            camera_id_mapping[camera_name] = 0
        else:
            # Convert Camera_0001 -> 1, Camera_0003 -> 3, etc.
            cam_num = int(camera_name.split('_')[1])
            camera_id_mapping[camera_name] = cam_num
    
    print(f"[PERSPECTIVE] Camera ID mapping: {camera_id_mapping}")

    # Perspective transform initialize (if available)
    try:
        calibrations_perspective = get_calibration_position(scene)
        print(f"[PERSPECTIVE] Found {len(calibrations_perspective)} calibration files")
        
        perspective_transforms = []
        for camera_name in camera_names:
            if len(calibrations_perspective) > 0:
                # Use the main calibration file for all cameras
                perspective_transforms.append(PerspectiveTransform(calibrations_perspective[0]))
                print(f"[PERSPECTIVE] Initialized for {camera_name}")
            else:
                perspective_transforms.append(None)
                print(f"[PERSPECTIVE] Warning: No calibration for {camera_name}")
                
    except Exception as e:
        print(f"[PERSPECTIVE] Warning: Could not load perspective transforms: {e}")
        perspective_transforms = [None] * len(camera_names)
 
    # Multi-camera tracker initialize
    clustering = Clustering(appearance_thresh=args['clt_appearance_thresh'], euc_thresh=args['clt_euclidean_thresh'],
                            match_thresh=0.8)
    mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.8, scene=scene)
    id_distributor = ID_Distributor()

    # Results lists for each camera
    results_lists = [[] for i in range(len(camera_names))]
    
    # Get total frames (minimum across all cameras)
    camera_handlers = [handler for handler in src_handlers if len(handler) == 4]
    total_frames = min([handler[3] for handler in camera_handlers])
    print(f"Processing {total_frames} frames across {len(camera_names)} cameras")
    
    # Load depth maps and calibration data for 3D processing
    depth_files = []
    camera_calibrations = []
    
    if enable_3d:
        depth_files = load_depth_maps(data_root, camera_names)
        camera_calibrations = load_calibration_data(data_root, camera_names)
    else:
        depth_files = [None] * len(camera_names)
        camera_calibrations = [None] * len(camera_names)

    # Load BEV map and parameters
    bev_dots_writer = None
    bev_boxes_writer = None
    map_img = None
    scale_factor = None
    tx = None
    ty = None
    map_h = None
    
    if enable_bev and enable_3d:
        map_img, scale_factor, tx, ty, map_h, map_w = load_bev_map_and_params(data_root, camera_calibrations)
        
        if map_img is not None:
            output_dir = f'output_videos/{scene}'
            os.makedirs(output_dir, exist_ok=True)
            bev_dots_writer, bev_boxes_writer = create_bev_writers(output_dir, scene, map_w, map_h, fps=30)
            
            # Print BEV scale information
            print(f"[BEV] Scale factor: {scale_factor:.2f} pixels/meter")
            print(f"[BEV] Expected person size: ~{0.5*scale_factor:.1f}x{0.3*scale_factor:.1f} pixels (0.5m x 0.3m)")
            if bev_scale_multiplier != 1.0:
                print(f"[BEV] With scale multiplier {bev_scale_multiplier}: ~{0.5*scale_factor*bev_scale_multiplier:.1f}x{0.3*scale_factor*bev_scale_multiplier:.1f} pixels")

    cur_frame = 1
    
    # Main processing loop
    while cur_frame <= total_frames:
        # Add frame limit for testing
        if cur_frame > 50:
            print(f"Reached frame limit of 500, stopping processing...")
            break
            
        imgs = []
        depth_maps = []
        all_cameras_have_frames = True
        
        # Process each camera
        for idx, (cap, writers, camera_name, frame_count) in enumerate(camera_handlers):
            # Read frame from video
            ret, img = cap.read()
            if not ret:
                print(f"End of video reached for {camera_name} at frame {cur_frame}")
                all_cameras_have_frames = False
                break
            
            # Load corresponding depth map
            depth_map = None
            if enable_3d and depth_files[idx] is not None:
                frame_num_str = str(cur_frame - 1).zfill(5)  # Frame numbering starts from 0
                depth_key = f"distance_to_image_plane_{frame_num_str}.png"
                
                if depth_key in depth_files[idx]:
                    depth_map = depth_files[idx][depth_key][:]
                else:
                    print(f"[Warning] Missing depth map: {depth_key} for {camera_name}")
            
            depth_maps.append(depth_map)

            # Run detection
            dets = detection(img, conf=conf_thres, iou=iou_thres, classes=[0,1,2,3], verbose=False)[0].boxes.data.cpu().numpy()

            # Calculate 3D positions if enabled
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

            # Run tracker for this camera
            img_path = f"{camera_name}_frame_{cur_frame:05d}"  # Synthetic path for compatibility
            online_targets, new_ratio = trackers[idx].update(dets, img, img_path, pose)
            
            if cur_frame <= 3:  # Debug info for first few frames
                print(f"[TRACKER] Frame {cur_frame}, {camera_name}: {len(online_targets)} tracks, pose: {'✅' if pose is not None else '❌'}")

            # Run perspective transform if available
            if perspective_transforms[idx] is not None:
                perspective_transforms[idx].run(trackers[idx], new_ratio)

            # Assign temporal global_id and initialize location for multi-camera tracking
            for t in trackers[idx].tracked_stracks:
                t.t_global_id = id_distributor.assign_id()
                # Initialize location if it doesn't exist (use centroid of bounding box)
                if not hasattr(t, 'location') or t.location is None:
                    centroid_x = (t.tlbr[0] + t.tlbr[2]) / 2
                    centroid_y = (t.tlbr[1] + t.tlbr[3]) / 2
                    t.location = [[centroid_x, centroid_y]]  # Double list to match expected format
                
            imgs.append(img)
        
        if not all_cameras_have_frames:
            break

        # Multi-camera tracking
        groups = clustering.update(trackers, cur_frame, scene)
        mc_tracker.update(trackers, groups)
        clustering.update_using_mctracker(trackers, mc_tracker)

        # Cluster refinement every 5 frames
        if cur_frame % 5 == 0:
            mc_tracker.refinement_clusters()

        # Update result lists
        try:
            # Create a simplified cam_ids structure for compatibility
            cam_ids_dict = {scene: camera_names}
            update_result_lists_testset(trackers, results_lists, cur_frame, cam_ids_dict, scene)
        except Exception as e:
            # Fallback: manually update results in MOT format
            for i, tracker in enumerate(trackers):
                for track in tracker.tracked_stracks:
                    if track.is_activated:
                        # MOT format: frame, id, left, top, width, height, conf, x, y, z
                        bbox = track.tlbr
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        result_line = f"{cur_frame},{track.global_id if hasattr(track, 'global_id') else track.track_id},{bbox[0]:.2f},{bbox[1]:.2f},{w:.2f},{h:.2f},{track.score:.2f},-1,-1,-1"
                        results_lists[i].append(result_line)

        # Write video outputs
        if args['write_vid']:
            if enable_3d:
                outputs = write_vids_3d_with_bev(trackers, imgs, camera_handlers, pose, _COLORS, mc_tracker, cur_frame,
                                               depth_maps=depth_maps, calibrations=camera_calibrations,
                                               map_img=map_img, scale_factor=scale_factor, tx=tx, ty=ty, map_h=map_h,
                                               bev_dots_writer=bev_dots_writer, bev_boxes_writer=bev_boxes_writer,
                                               bev_scale_multiplier=bev_scale_multiplier, debug_bev=debug_bev)
            else:
                # For 2D mode, create a simplified version
                for i, (tracker, img, (_, writers, camera_name, _)) in enumerate(zip(trackers, imgs, camera_handlers)):
                    # Simple 2D visualization (you may need to adapt this)
                    for track in tracker.tracked_stracks:
                        if track.is_activated:
                            bbox = track.tlbr.astype(int)
                            x1, y1, x2, y2 = bbox
                            track_id = track.global_id if hasattr(track, 'global_id') and track.global_id != -1 else track.track_id
                            color_rgb = _COLORS[track_id % len(_COLORS)]
                            color = (int(color_rgb[2] * 255), int(color_rgb[1] * 255), int(color_rgb[0] * 255))
                            
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, f'{track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    writers[0].write(img)

        if cur_frame % 50 == 0:  # Print progress every 50 frames
            print(f"Processed frame {cur_frame}/{total_frames}")
        
        cur_frame += 1
    
    # Cleanup
    finalize_cameras_with_bev(camera_handlers, bev_dots_writer, bev_boxes_writer)
    print(f"[CLEANUP] Released {len(camera_handlers)} cameras")
    
    # Close depth files
    if enable_3d:
        for depth_file in depth_files:
            if depth_file is not None:
                depth_file.close()
        print(f"[CLEANUP] Closed {len([f for f in depth_files if f is not None])} depth files")

    # Save results
    try:
        # Create result paths for the new format
        result_paths_list = [f"output_videos/{scene}/{camera_name}_results.txt" for camera_name in camera_names]
        write_results_testset(results_lists, result_paths_list)
        print(f"[RESULTS] Saved results for {len(camera_names)} cameras")
    except Exception as e:
        print(f"[RESULTS] Warning: Could not save results in standard format: {e}")
        # Save results manually
        for i, (results, camera_name) in enumerate(zip(results_lists, camera_names)):
            result_path = f"output_videos/{scene}/{camera_name}_results.txt"
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, 'w') as f:
                for result in results:
                    f.write(f"{result}\n")
            print(f"[RESULTS] Manually saved {len(results)} results for {camera_name}")
    
    print('[SYSTEM] Processing completed successfully!')
    if enable_bev and map_img is not None:
        print(f'[BEV] Bird\'s Eye View videos saved:')
        print(f'  - Dots view: output_videos/{scene}/{scene}_bev_dots.mp4')
        print(f'  - 3D Boxes view: output_videos/{scene}/{scene}_bev_boxes.mp4')


if __name__ == '__main__':
    args = {
        'max_batch_size': 32,
        'track_buffer': 150,
        'with_reid': True,
        'sct_appearance_thresh': 0.4,
        'sct_euclidean_thresh': 0.1,
        'clt_appearance_thresh': 0.35,
        'clt_euclidean_thresh': 0.3,
        'mct_appearance_thresh': 0.4,
        'frame_rate': 30,
        'write_vid': True,
    }

    parsed_args = make_parser()
    scene = parsed_args.scene
    enable_3d = parsed_args.enable_3d
    data_root = parsed_args.data_root
    bev_scale_multiplier = parsed_args.bev_scale_multiplier
    debug_bev = parsed_args.debug_bev
    
    # Setup logging if requested
    log_file = None
    if parsed_args.save_logs:
        log_file = parsed_args.log_file
        print(f"Logging enabled: {log_file}")

    print(f"Starting multi-camera tracking for scene: {scene}")
    print(f"Data root: {data_root}")
    print(f"3D mode: {'Enabled' if enable_3d else 'Disabled'}")
    print(f"BEV mode: {'Enabled (Dual Views)' if enable_3d else 'Disabled'} (automatically enabled with 3D)")
    if enable_3d:
        print(f"BEV scale multiplier: {bev_scale_multiplier}")
        print(f"BEV debug: {'Enabled' if debug_bev else 'Disabled'}")
    
    run(args=args, conf_thres=0.1, iou_thres=0.45, data_root=data_root, 
        scene=scene, enable_3d=enable_3d, log_file=log_file, 
        bev_scale_multiplier=bev_scale_multiplier, debug_bev=debug_bev)