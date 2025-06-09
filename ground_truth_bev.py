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

# Import 3D bounding box functions
from tools.utils_new import create_3d_bbox, draw_3d_box, project_3d_box_to_bev, visualize

from tools.utils_mod import SCENE_ID_MAPPING,CLASS_ID_MAPPING,_COLORS

import cv2
import os
import time
import numpy as np
import argparse
import h5py
import math
import glob
from pathlib import Path





def make_parser():
    parser = argparse.ArgumentParser(description="Run Online MTPC System")
    parser.add_argument("-s", "--scene", type=str, default="Warehouse_013", help="scene name to inference")
    parser.add_argument("--enable_3d", action="store_true", help="Enable 3D bounding box mode")
    parser.add_argument("--enable_bev", action="store_true", help="Enable Bird's Eye View visualization")
    parser.add_argument("--show_gt", action="store_true", help="Show ground truth on BEV")
    parser.add_argument("--data_root", type=str, default="data/Warehouse_013", help="Root directory containing videos and depth maps")
    parser.add_argument("--gt_path", type=str, default="data/Warehouse_013/ground_truth.json", help="Path to ground truth JSON")
    return parser.parse_args()


def get_video_readers_and_writers(data_root, scene_name, enable_3d=False, enable_bev=False):
    """
    Get video readers from MP4 files and create corresponding writers
    
    Args:
        data_root: Root directory containing videos and depth maps
        scene_name: Name of the scene (e.g., "Warehouse_002")
        enable_3d: Whether to create 3D visualization writers
        enable_bev: Whether to create BEV visualization writers
    
    Returns:
        List of [video_reader, writers, camera_name, frame_count] for each camera
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
        camera_name = os.path.splitext(os.path.basename(video_file))[0]  # e.g., "Camera_0003"
        
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
        
        if enable_bev:
            # BEV video writer placeholder (will be set up later when we know map dimensions)
            writers.append(None)
        
        src_handlers.append([cap, writers, camera_name, frame_count])
    
    return src_handlers


def load_ground_truth(gt_path):
    """Load ground truth data from JSON file"""
    try:
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        print(f"[GT] Loaded ground truth from {gt_path}")
        return gt_data
    except Exception as e:
        print(f"[GT] Failed to load ground truth: {e}")
        return None


def load_depth_maps(data_root, camera_names):
    """
    Load depth map files for each camera
    
    Args:
        data_root: Root directory containing depth maps
        camera_names: List of camera names (e.g., ["Camera_0003", "Camera_0005"])
    
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
        camera_names: List of camera names (e.g., ["Camera_0003", "Camera_0005"])
    
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
                # formatted_name = f"Camera_{cam_num.zfill(4)}"
                formatted_name = f"Camera_{cam_num.zfill(1)}"
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


def world_to_bev_coordinates(world_coords, scale_factor, tx, ty, map_height):
    """Convert 3D world coordinates to BEV map pixel coordinates"""
    world_x, world_y = world_coords[0], world_coords[1]
    
    # Convert world coordinates to map coordinates
    map_x = (world_x + tx) * scale_factor
    map_y = (world_y + ty) * scale_factor
    
    # Convert to image coordinates (flip Y axis)
    pixel_u = int(map_x)
    pixel_v = int(map_height - map_y)  # Flip Y coordinate
    
    return [pixel_u, pixel_v]


def create_oriented_bbox_corners(center, scale, rotation):
    """Create 3D bounding box corners with orientation"""
    # Create local bounding box corners (centered at origin)
    w, l, h = scale[0] / 2, scale[1] / 2, scale[2] / 2
    
    # Define 8 corners of the box
    corners = np.array([
        [-w, -l, -h],  # bottom face
        [w, -l, -h],
        [w, l, -h],
        [-w, l, -h],
        [-w, -l, h],   # top face
        [w, -l, h],
        [w, l, h],
        [-w, l, h]
    ])
    
    # Apply rotation (only Z rotation for now, since most objects rotate around Z-axis)
    rz = rotation[2]  # Z rotation (yaw)
    cos_z, sin_z = np.cos(rz), np.sin(rz)
    
    # Rotation matrix for Z-axis
    R_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])
    
    # Apply rotation to all corners
    rotated_corners = np.dot(corners, R_z.T)
    
    # Translate to world position
    world_corners = rotated_corners + np.array(center)
    
    return world_corners


def draw_oriented_box_on_bev(bev_img, center, scale, rotation, scale_factor, tx, ty, map_height, 
                           color=(0, 255, 0), thickness=2, label="", object_type="Person"):
    """Draw oriented 3D bounding box on BEV map"""
    try:
        # Create oriented bounding box corners
        corners_3d = create_oriented_bbox_corners(center, scale, rotation)
        
        # Convert all corners to BEV coordinates
        bev_corners = []
        for corner in corners_3d:
            bev_coord = world_to_bev_coordinates(corner, scale_factor, tx, ty, map_height)
            bev_corners.append(bev_coord)
        
        bev_corners = np.array(bev_corners, dtype=np.int32)
        
        # Check if corners are within image bounds
        h_img, w_img = bev_img.shape[:2]
        valid_corners = []
        for corner in bev_corners:
            if 0 <= corner[0] < w_img and 0 <= corner[1] < h_img:
                valid_corners.append(corner)
        
        if len(valid_corners) < 2:  # Not enough visible corners
            return bev_img
        
        bev_corners = np.array(bev_corners)
        
        # Draw the bottom face (footprint)
        bottom_face = bev_corners[:4]  # First 4 corners are bottom face
        
        # Make sure corners are within bounds for drawing
        bottom_face = np.clip(bottom_face, [0, 0], [w_img-1, h_img-1])
        
        cv2.polylines(bev_img, [bottom_face], isClosed=True, color=color, thickness=thickness)
        
        # Draw direction arrow (from center to front)
        center_bev = world_to_bev_coordinates(center, scale_factor, tx, ty, map_height)
        
        # Calculate front point based on rotation
        front_offset = 0.5  # meters
        front_x = center[0] + front_offset * np.cos(rotation[2])
        front_y = center[1] + front_offset * np.sin(rotation[2])
        front_bev = world_to_bev_coordinates([front_x, front_y, center[2]], scale_factor, tx, ty, map_height)
        
        # Draw arrow if both points are within bounds
        if (0 <= center_bev[0] < w_img and 0 <= center_bev[1] < h_img and
            0 <= front_bev[0] < w_img and 0 <= front_bev[1] < h_img):
            cv2.arrowedLine(bev_img, tuple(center_bev), tuple(front_bev), color, thickness)
        
        # Draw center point
        if 0 <= center_bev[0] < w_img and 0 <= center_bev[1] < h_img:
            cv2.circle(bev_img, tuple(center_bev), 3, color, -1)
        
        # Draw label
        if label and 0 <= center_bev[0] < w_img and 0 <= center_bev[1] < h_img:
            cv2.putText(bev_img, label, (center_bev[0] + 5, center_bev[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    except Exception as e:
        print(f"[BEV] Error drawing oriented box: {e}")
    
    return bev_img


def draw_ground_truth_on_bev(bev_img, gt_data, frame_idx, scale_factor, tx, ty, map_height):
    """Draw ground truth objects on BEV map"""
    frame_key = str(frame_idx)
    if frame_key not in gt_data:
        return bev_img
    
    # Define colors for different object types
    object_colors = {
        "Person": (0, 255, 0),        # Green
        "Forklift": (255, 0, 0),      # Blue
        "NovaCarter": (0, 165, 255),  # Orange
        "Transporter": (255, 255, 0), # Cyan
        "default": (128, 128, 128)    # Gray
    }
    
    for obj in gt_data[frame_key]:
        obj_type = obj.get("object type", "unknown")
        obj_id = obj.get("object id", -1)
        location = obj.get("3d location", [0, 0, 0])
        scale = obj.get("3d bounding box scale", [1, 1, 1])
        rotation = obj.get("3d bounding box rotation", [0, 0, 0])
        
        # Get color for this object type
        color = object_colors.get(obj_type, object_colors["default"])
        
        # Create label
        label = f"GT_{obj_type}_{obj_id}"
        
        # Draw oriented bounding box
        bev_img = draw_oriented_box_on_bev(
            bev_img, location, scale, rotation, scale_factor, tx, ty, map_height,
            color=color, thickness=2, label=label, object_type=obj_type
        )
    
    return bev_img


def draw_3d_box_on_bev(bev_img, corners_3d_world, scale_factor, tx, ty, map_height, color=(0, 255, 0), thickness=2):
    """Draw 3D bounding box projected onto BEV map (for tracking results)"""
    if corners_3d_world is None:
        return bev_img
    
    # Convert all corners to BEV coordinates
    bev_corners = []
    for corner in corners_3d_world:
        bev_coord = world_to_bev_coordinates(corner, scale_factor, tx, ty, map_height)
        bev_corners.append(bev_coord)
    
    bev_corners = np.array(bev_corners, dtype=np.int32)
    
    # Check bounds
    h_img, w_img = bev_img.shape[:2]
    bev_corners = np.clip(bev_corners, [0, 0], [w_img-1, h_img-1])
    
    # Draw the bottom face of the box (feet level)
    bottom_face = bev_corners[:4]  # First 4 corners are bottom face
    cv2.polylines(bev_img, [bottom_face], isClosed=True, color=color, thickness=thickness)
    
    # Draw center point
    center_3d = np.mean(corners_3d_world, axis=0)
    center_bev = world_to_bev_coordinates(center_3d, scale_factor, tx, ty, map_height)
    
    if 0 <= center_bev[0] < w_img and 0 <= center_bev[1] < h_img:
        cv2.circle(bev_img, tuple(center_bev), 4, color, -1)
    
    return bev_img


def write_results_unified(result_lists, scene_name, output_dir):
    """
    Write results in the unified format:
    <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
    
    Args:
        result_lists: List of results from all cameras
        scene_name: Name of the scene (e.g., "Warehouse_002")
        output_dir: Output directory for results
    """
    # Get scene ID from mapping
    scene_id = SCENE_ID_MAPPING.get(scene_name, 0)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{scene_name}_tracking_results.txt")
    
    # Collect all results and sort by frame_id and object_id
    all_results = []
    for result_list in result_lists:
        all_results.extend(result_list)
    
    # Sort by frame_id, then by object_id
    all_results.sort(key=lambda x: (x['frame_id'], x['object_id']))
    
    # Write results in the specified format
    with open(result_path, 'w') as f:
        for result in all_results:
            # Extract values
            class_id = result['class_id']  # Person = 0
            object_id = result['object_id']
            frame_id = result['frame_id']
            x, y, z = result['world_coords']
            width, length, height = result['dimensions']
            yaw = result['yaw']
            
            # Write in the specified format
            line = f"{scene_id} {class_id} {object_id} {frame_id} {x:.2f} {y:.2f} {z:.2f} {width:.2f} {length:.2f} {height:.2f} {yaw:.2f}\n"
            f.write(line)
    
    print(f"[RESULTS] Saved {len(all_results)} tracking results to {result_path}")


def update_result_lists_with_3d(trackers, result_lists, frame_id, depth_maps, calibrations):
    """
    Update result lists with 3D information for unified output format
    
    Args:
        trackers: List of trackers for each camera
        result_lists: List to store results
        frame_id: Current frame number (0-based)
        depth_maps: Depth maps for each camera
        calibrations: Camera calibration data
    """
    for tracker_idx, (tracker, depth_map, calib) in enumerate(zip(trackers, depth_maps, calibrations)):
        for track in tracker.tracked_stracks:
            if not track.is_activated or track.global_id < 0:
                continue
            
            # Default values
            world_coords = [0.0, 0.0, 0.0]
            dimensions = [0.6, 0.6, 1.8]  # Default person dimensions (width, length, height)
            yaw = 0.0  # Default orientation
            
            # Try to get 3D information if depth map and calibration are available
            if depth_map is not None and calib is not None:
                try:
                    bbox = track.tlbr.astype(int)
                    x1, y1, x2, y2 = bbox
                    bbox_2d = [x1, y1, x2, y2]
                    
                    # Get camera parameters
                    K = np.array(calib['intrinsicMatrix'])
                    E = np.array(calib['extrinsicMatrix'])
                    R = E[:, :3]
                    t = E[:, 3].reshape((3, 1))
                    
                    # Create 3D bounding box
                    corners_3d_world, center_3d_world, bbox_dimensions = create_3d_bbox(
                        bbox_2d, depth_map, K, R, t, depth_scale=1000.0)
                    
                    if center_3d_world is not None and bbox_dimensions is not None:
                        world_coords = center_3d_world.tolist()
                        # Convert dimensions to [width, length, height] format
                        # bbox_dimensions typically comes as [width, height, depth]
                        dimensions = [
                            float(bbox_dimensions[0]),  # width
                            float(bbox_dimensions[2]),  # length (depth)
                            float(bbox_dimensions[1])   # height
                        ]
                        
                        # Estimate yaw from movement direction if possible
                        # For now, use default 0.0
                        yaw = 0.0
                        
                except Exception as e:
                    print(f"[3D] Error processing track {track.global_id}: {e}")
                    # Keep default values
            
            # Create result entry
            result = {
                'class_id': CLASS_ID_MAPPING['Person'],  # Currently only tracking people
                'object_id': track.global_id,
                'frame_id': frame_id,
                'world_coords': world_coords,
                'dimensions': dimensions,
                'yaw': yaw,
                'camera_id': tracker_idx,
                'bbox_2d': track.tlbr.tolist(),
                'score': track.score
            }
            
            result_lists[tracker_idx].append(result)


def write_vids_with_bev_and_gt(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, cur_frame, 
                              depth_maps=None, calibrations=None, map_img=None, bev_writer=None,
                              scale_factor=None, tx=None, ty=None, map_h=None, enable_3d=False, 
                              enable_bev=False, gt_data=None, show_gt=False):
    """Enhanced version that includes BEV visualization with ground truth"""
    outputs = []
    
    # Get feature information from mc_tracker
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
        
        # Draw ground truth first (so tracking results appear on top)
        if show_gt and gt_data is not None:
            bev_frame = draw_ground_truth_on_bev(bev_frame, gt_data, cur_frame - 1, 
                                               scale_factor, tx, ty, map_h)
    
    for i, (tracker, img, (_, writers, camera_name, _), depth_map, calib) in enumerate(zip(trackers, imgs, src_handlers, depth_maps, calibrations)):
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
                            
                            # Draw tracking results on BEV map (in different color from GT)
                            if enable_bev and bev_frame is not None and scale_factor and tx and ty and map_h:
                                # Use brighter colors for tracking results to distinguish from GT
                                track_color = (int(color[0] * 0.7), int(color[1] * 0.7), 255)  # Add blue tint
                                
                                bev_frame = draw_3d_box_on_bev(
                                    bev_frame, corners_3d_world, scale_factor, tx, ty, map_h,
                                    color=track_color, thickness=2)
                                
                                # Add track ID label on BEV
                                center_bev = world_to_bev_coordinates(center_3d_world, scale_factor, tx, ty, map_h)
                                h_img, w_img = bev_frame.shape[:2]
                                if 0 <= center_bev[0] < w_img and 0 <= center_bev[1] < h_img:
                                    cv2.putText(bev_frame, f'T_{track_id}', 
                                              (center_bev[0] + 10, center_bev[1] - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2)
                            
                    except Exception as e:
                        print(f"Error creating 3D bbox for track {track_id} in {camera_name}: {e}")
        
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
    
    # Write BEV frame with legend
    if enable_bev and bev_writer is not None and bev_frame is not None:
        # Add legend to BEV frame
        if show_gt:
            legend_y = 30
            cv2.putText(bev_frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            legend_y += 25
            cv2.putText(bev_frame, "GT_Person (Green)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            legend_y += 20
            cv2.putText(bev_frame, "GT_Forklift (Blue)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            legend_y += 20
            cv2.putText(bev_frame, "GT_NovaCarter (Orange)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            legend_y += 20
            cv2.putText(bev_frame, "Tracking Results (Blue-tinted)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(bev_frame, f"Frame: {cur_frame}", (bev_frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        bev_writer.write(bev_frame)
    
    return outputs


def finalize_cameras(src_handlers):
    """Release video readers and writers"""
    for cap, writers, camera_name, _ in src_handlers:
        cap.release()
        for writer in writers:
            if writer is not None:
                writer.release()
        print(f"Released {camera_name}")


def run(args, conf_thres, iou_thres, data_root, scene, enable_3d=False, enable_bev=False, show_gt=False, gt_path=None):
    # Load ground truth data
    gt_data = None
    if show_gt and gt_path:
        gt_data = load_ground_truth(gt_path)
    
    print(f"[SYSTEM] Starting single-camera tracking for scene: {scene}")
    print(f"[SYSTEM] Data root: {data_root}")
    print(f"[SYSTEM] 3D mode: {'Enabled' if enable_3d else 'Disabled'}")
    print(f"[SYSTEM] BEV mode: {'Enabled' if enable_bev else 'Disabled'}")
    print(f"[SYSTEM] Ground Truth: {'Enabled' if show_gt else 'Disabled'}")
    
    # Detection model initialize
    detection = YOLO('runs/best.pt')
    print('[DETECTION] Model loaded:', detection)
        
    # Pose estimation initialize
    config_file = 'mmpose/configs/body_2d_keypoint/rtmpose/crowdpose/rtmpose-m_8xb64-210e_crowdpose-256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth'
    pose = init_model(config_file, checkpoint_file, device='cuda:0')
    print(f"[POSE] Model loaded successfully with {pose.dataset_meta.get('num_joints', 'unknown')} joints")

    # Get video readers and writers for all cameras
    src_handlers = get_video_readers_and_writers(data_root, scene, enable_3d, enable_bev)
    camera_names = [handler[2] for handler in src_handlers]
    print(f"[CAMERAS] Found cameras: {camera_names}")
    
    # Initialize trackers for each camera
    trackers = []
    for i in range(len(src_handlers)):
       trackers.append(BoTSORT(track_buffer=args['track_buffer'], max_batch_size=args['max_batch_size'], 
                            appearance_thresh=args['sct_appearance_thresh'], euc_thresh=args['sct_euclidean_thresh']))
    print(f"[TRACKERS] Initialized for {len(trackers)} cameras")

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
        perspective_transforms = [None] * len(src_handlers)
 
    # Multi-camera tracker initialize
    clustering = Clustering(appearance_thresh=args['clt_appearance_thresh'], euc_thresh=args['clt_euclidean_thresh'],
                            match_thresh=0.8)
    mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.8, scene=scene)
    id_distributor = ID_Distributor()

    # Results lists for each camera
    results_lists = [[] for i in range(len(src_handlers))]
    
    # Get total frames (minimum across all cameras)
    total_frames = min([handler[3] for handler in src_handlers])
    print(f"Processing {total_frames} frames across {len(src_handlers)} cameras")
    
    # Load depth maps and calibration data
    depth_files = []
    camera_calibrations = []
    map_img = None
    bev_writer = None
    scale_factor = None
    tx = None
    ty = None
    map_h = None
    
    if enable_3d or enable_bev:
        depth_files = load_depth_maps(data_root, camera_names)
        camera_calibrations = load_calibration_data(data_root, camera_names)
        
        # Setup BEV visualization if enabled
        if enable_bev:
            try:
                map_path = os.path.join(data_root, "map.png")
                map_img = cv2.imread(map_path)
                if map_img is not None:
                    map_h, map_w = map_img.shape[:2]
                    
                    # Get BEV parameters from first available calibration
                    for calib in camera_calibrations:
                        if calib is not None:
                            scale_factor = calib.get("scaleFactor", 14.420706776686613)
                            tx = calib.get("translationToGlobalCoordinates", {}).get("x", 76.54132562910155)
                            ty = calib.get("translationToGlobalCoordinates", {}).get("y", 41.46934389588847)
                            break
                    
                    print(f"[BEV] Map loaded: {map_w}x{map_h}, scale: {scale_factor}, translation: ({tx}, {ty})")
                    
                    # Create BEV video writer
                    suffix = "_with_gt" if show_gt else ""
                    bev_output_path = f"output_videos/{scene}/bev_tracking{suffix}.mp4"
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
        depth_files = [None] * len(src_handlers)
        camera_calibrations = [None] * len(src_handlers)

    cur_frame = 1
    
    # Main processing loop
    while cur_frame <= total_frames:
        # Add frame limit for testing
        if cur_frame > 2000:
            print(f"Reached frame limit of 1000, stopping processing...")
            break
            
        imgs = []
        depth_maps = []
        all_cameras_have_frames = True
        
        # Process each camera
        for idx, (cap, writers, camera_name, frame_count) in enumerate(src_handlers):
            # Read frame from video
            ret, img = cap.read()
            if not ret:
                print(f"End of video reached for {camera_name} at frame {cur_frame}")
                all_cameras_have_frames = False
                break
            
            # Load corresponding depth map
            depth_map = None
            if (enable_3d or enable_bev) and depth_files[idx] is not None:
                frame_num_str = str(cur_frame - 1).zfill(5)  # Frame numbering starts from 0
                depth_key = f"distance_to_image_plane_{frame_num_str}.png"
                
                if depth_key in depth_files[idx]:
                    depth_map = depth_files[idx][depth_key][:]
                else:
                    print(f"[Warning] Missing depth map: {depth_key} for {camera_name}")
            
            depth_maps.append(depth_map)

            # Run detection
            dets = detection(img, conf=conf_thres, iou=iou_thres, classes=[0,1,2,3], verbose=False)[0].boxes.data.cpu().numpy()

            # Run tracker for this camera
            img_path = f"{camera_name}_frame_{cur_frame:05d}"  # Synthetic path for compatibility
            online_targets, new_ratio = trackers[idx].update(dets, img, img_path, pose)
            
            if cur_frame <= 3:  # Debug info for first few frames
                print(f"[TRACKER] Frame {cur_frame}, {camera_name}: {len(online_targets)} tracks, pose: {'✅' if pose is not None else '❌'}")

            # Run perspective transform if available
            if perspective_transforms[idx] is not None:
                perspective_transforms[idx].run(trackers[idx], new_ratio)

            # Assign temporal global_id for multi-camera tracking
            for t in trackers[idx].tracked_stracks:
                t.t_global_id = id_distributor.assign_id()
                # Initialize location if it doesn't exist
                if not hasattr(t, 'location') or t.location is None:
                    centroid_x = (t.tlbr[0] + t.tlbr[2]) / 2
                    centroid_y = (t.tlbr[1] + t.tlbr[3]) / 2
                    t.location = [[centroid_x, centroid_y]]
                
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

        # Update result lists with 3D information
        update_result_lists_with_3d(trackers, results_lists, cur_frame - 1, depth_maps, camera_calibrations)

        # Write video outputs
        if args['write_vid']:
            outputs = write_vids_with_bev_and_gt(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, cur_frame,
                                               depth_maps=depth_maps, calibrations=camera_calibrations,
                                               map_img=map_img, bev_writer=bev_writer,
                                               scale_factor=scale_factor, tx=tx, ty=ty, map_h=map_h,
                                               enable_3d=enable_3d, enable_bev=enable_bev, 
                                               gt_data=gt_data, show_gt=show_gt)

        if cur_frame % 50 == 0:  # Print progress every 50 frames
            print(f"Processed frame {cur_frame}/{total_frames}")
        
        cur_frame += 1
    
    # Cleanup
    finalize_cameras(src_handlers)
    print(f"[CLEANUP] Released {len(src_handlers)} cameras")
    
    # Close depth files
    if enable_3d or enable_bev:
        for depth_file in depth_files:
            if depth_file is not None:
                depth_file.close()
        print(f"[CLEANUP] Closed {len([f for f in depth_files if f is not None])} depth files")

    if bev_writer is not None:
        bev_writer.release()
        print("[CLEANUP] BEV video writer released")

    # Save results in unified format
    output_dir = f"output_videos/{scene}"
    write_results_unified(results_lists, scene, output_dir)
    
    print('[SYSTEM] Processing completed successfully!')


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
    enable_bev = parsed_args.enable_bev
    show_gt = parsed_args.show_gt
    data_root = parsed_args.data_root
    gt_path = parsed_args.gt_path

    print(f"Starting tracking for scene: {scene}")
    print(f"Data root: {data_root}")
    print(f"3D mode: {'Enabled' if enable_3d else 'Disabled'}")
    print(f"BEV mode: {'Enabled' if enable_bev else 'Disabled'}")
    print(f"Ground Truth: {'Enabled' if show_gt else 'Disabled'}")
    
    run(args=args, conf_thres=0.1, iou_thres=0.45, data_root=data_root, 
        scene=scene, enable_3d=enable_3d, enable_bev=enable_bev, show_gt=show_gt, gt_path=gt_path)