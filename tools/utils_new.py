import numpy as np
import h5py

import cv2
import numpy as np

def pixel_depth_to_world_map(u, v, depth_mm, K, R, t):
    """
    Convert pixel coordinates and depth to world coordinates and map position
    
    Args:
        u, v: Pixel coordinates in the image
        depth_mm: Depth value in millimeters
        K: Camera intrinsic matrix
        R: Rotation matrix (from camera to world)
        t: Translation vector (from camera to world)
    
    Returns:
        world_coords: (x, y, z) in world coordinates (meters)
        map_coords: (x, y) in map image coordinates (pixels)
    """
    # Convert depth from mm to meters (if needed)
    depth_m = depth_mm / 1000.0
    
    # 1. Convert pixel to 3D point in camera coordinates
    K_inv = np.linalg.inv(K)
    uv1 = np.array([[u], [v], [1]])
    
    # Scale by depth to get 3D point in camera coordinates (meters)
    point_camera = depth_m * (K_inv @ uv1)
    
    # 2. Transform from camera to world coordinates
    # The correct way to transform a point from camera to world:
    point_world = (R.T @ point_camera) - (R.T @ t)
    
    # Get world coordinates in meters
    wx, wy, wz = point_world.flatten()
    
    # 3. Project to map coordinates
    # Assuming ground plane is z=0, we can just use x,y
    # Apply scaling and translation to convert world coordinates to map pixels
    map_x = int((wx + tx) * scale_factor)
    map_y = map_h - int((wy + ty) * scale_factor)  # Flip y-axis
    
    return (wx, wy, wz), (map_x, map_y)


def print_hdf5_structure(name, obj):
    print(f"Found object: {name}, Type: {type(obj)}")
    


def create_3d_bbox(bbox_2d, depth_map, K, R, t, depth_scale=1000.0):
    """
    Create a 3D bounding box from a 2D bounding box and depth map
    
    Args:
        bbox_2d (list): [x1, y1, x2, y2] in pixel coordinates
        depth_map (numpy array): Depth map of the scene
        K (numpy array): Camera intrinsic matrix (3x3)
        R (numpy array): Rotation matrix (camera to world) (3x3)
        t (numpy array): Translation vector (camera to world) (3x1)
        depth_scale (float): Scale factor to convert depth map values to meters
    
    Returns:
        corners_3d_world (numpy array): 8 corners of 3D bounding box in world coordinates
        center_3d_world (numpy array): Center of 3D bounding box in world coordinates
        dimensions (numpy array): Dimensions of 3D bounding box (width, height, depth)
    """
    # Extract 2D bounding box parameters
    x1, y1, x2, y2 = bbox_2d
    
    # Calculate center and dimensions of 2D box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width_2d = x2 - x1
    height_2d = y2 - y1
    
    # Sample depth values within the bounding box
    # Use a grid or key points to get reliable depth values
    box_depth_values = []
    grid_size = 5  # Sample a 5x5 grid within the box
    
    for i in range(grid_size):
        for j in range(grid_size):
            sample_x = int(x1 + (width_2d * i) / (grid_size - 1))
            sample_y = int(y1 + (height_2d * j) / (grid_size - 1))
            
            if 0 <= sample_x < depth_map.shape[1] and 0 <= sample_y < depth_map.shape[0]:
                depth_value = depth_map[sample_y, sample_x]
                if depth_value > 0:  # Avoid invalid depth values
                    box_depth_values.append(depth_value)
    
    # Get median depth (more robust than mean)
    if not box_depth_values:
        return None, None, None  # No valid depth values
    
    median_depth = np.median(box_depth_values) / depth_scale  # Convert to meters
    
    # Calculate the center point in 3D camera coordinates
    center_2d = np.array([[center_x], [center_y], [1.0]])
    K_inv = np.linalg.inv(K)
    center_3d_camera = median_depth * (K_inv @ center_2d)
    
    # Convert center from camera to world coordinates
    center_3d_world = (R.T @ center_3d_camera) - (R.T @ t)
    center_3d_world = center_3d_world.flatten()
    
    # Estimate physical dimensions based on 2D box and depth
    # This is a simplification and can be improved with actual object dimensions
    # or constraints based on object class
    width_3d = width_2d * median_depth / K[0, 0]   # fx is K[0, 0]
    height_3d = height_2d * median_depth / K[1, 1]  # fy is K[1, 1]
    
    # Estimate depth dimension (can be improved)
    # For simple cases, we can use the width or a fixed ratio
    depth_3d = width_3d * 0.8  # Assuming depth is 80% of width (adjust for your objects)
    
    dimensions = np.array([width_3d, height_3d, depth_3d])
    
    # Create 8 corners of the 3D box in camera coordinates
    # Format: (x, y, z) relative to the center
    half_width = width_3d / 2
    half_height = height_3d / 2
    half_depth = depth_3d / 2
    
    corners_3d_camera = np.array([
        [-half_width, -half_height, -half_depth],  # Back-bottom-left
        [half_width, -half_height, -half_depth],   # Back-bottom-right
        [half_width, half_height, -half_depth],    # Back-top-right
        [-half_width, half_height, -half_depth],   # Back-top-left
        [-half_width, -half_height, half_depth],   # Front-bottom-left
        [half_width, -half_height, half_depth],    # Front-bottom-right
        [half_width, half_height, half_depth],     # Front-top-right
        [-half_width, half_height, half_depth]     # Front-top-left
    ])
    
    # Transform corners to world coordinates
    corners_3d_world = []
    for corner in corners_3d_camera:
        # Add corner to center in camera coordinates
        corner_camera = center_3d_camera.flatten() + corner
        # Convert to world coordinates
        corner_world = (R.T @ corner_camera.reshape(3, 1)) - (R.T @ t)
        corners_3d_world.append(corner_world.flatten())
    
    corners_3d_world = np.array(corners_3d_world)
    
    return corners_3d_world, center_3d_world, dimensions

def draw_3d_box(frame, corners_3d_world, K, R, t, color=(0, 255, 0), thickness=2):
    """
    Draw a 3D bounding box on the image
    
    Args:
        frame (numpy array): Image to draw on
        corners_3d_world (numpy array): 8 corners of 3D bounding box in world coordinates
        K (numpy array): Camera intrinsic matrix
        R (numpy array): Rotation matrix (world to camera)
        t (numpy array): Translation vector (world to camera)
        color (tuple): RGB color for the lines
        thickness (int): Line thickness
    
    Returns:
        frame (numpy array): Image with 3D box drawn
    """
    # Define the edges of the 3D bounding box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    
    # Project 3D corners to 2D image points
    corners_2d = []
    for corner in corners_3d_world:
        # Convert corner from world to camera coordinates
        corner_camera = R @ corner.reshape(3, 1) + t
        
        # Project to image plane
        x = K[0, 0] * corner_camera[0, 0] / corner_camera[2, 0] + K[0, 2]
        y = K[1, 1] * corner_camera[1, 0] / corner_camera[2, 0] + K[1, 2]
        
        corners_2d.append((int(x), int(y)))
    
    # Draw the edges
    for i, j in edges:
        cv2.line(frame, corners_2d[i], corners_2d[j], color, thickness)
    
    return frame

def project_3d_box_to_bev(corners_3d_world, map_img, scale_factor, tx, ty, map_h,
                           color=(0, 0, 255), thickness=2):
    """
    Project 3D bounding box to bird's eye view map
    
    Args:
        corners_3d_world (numpy array): 8 corners of 3D bounding box in world coordinates
        map_img (numpy array): Bird's eye view map image
        scale_factor (float): Scale factor for converting world to map coordinates
        tx, ty (float): Translation offsets
        map_h (int): Height of the map image
        color (tuple): RGB color for the box
        thickness (int): Line thickness
    
    Returns:
        map_img (numpy array): Map image with 3D box projected to BEV
    """
    # Get the bottom face corners (assuming corners 0,1,5,4 are the bottom face)
    # This may need to be adjusted based on your coordinate system
    bottom_face = [corners_3d_world[0], corners_3d_world[1], 
                  corners_3d_world[5], corners_3d_world[4]]
    
    # Convert corners from world to map coordinates
    map_points = []
    for corner in bottom_face:
        map_x = int((corner[0] + tx) * scale_factor)
        map_y = map_h - int((corner[1] + ty) * scale_factor)
        map_points.append((map_x, map_y))
    
    # Draw the bottom face as a polygon on the map
    map_points = np.array(map_points, np.int32)
    map_points = map_points.reshape((-1, 1, 2))
    cv2.polylines(map_img, [map_points], True, color, thickness)
    
    # Optionally, fill the polygon with a transparent color
    overlay = map_img.copy()
    cv2.fillPoly(overlay, [map_points], color)
    cv2.addWeighted(overlay, 0.3, map_img, 0.7, 0, map_img)
    
    return map_img

# def visualize(dets, img, colors, cur_frame):
#     """
#     Visualize detections on image with bounding boxes and track information
    
#     Args:
#         dets: List of detections, each containing [x1, y1, x2, y2, score, track_id, len_feats]
#         img: Input image to draw on
#         colors: Color palette for different track IDs
#         cur_frame: Current frame number
    
#     Returns:
#         img: Image with visualizations drawn
#     """
#     m = 2
#     for obj in dets:
#         score = obj[4]
#         track_id = int(obj[5])
#         len_feats = obj[6]
#         x0, y0, x1, y1 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
#         center_x = (x0 + x1) // 2
#         center_y = (y0 + y1) // 2
        
#         # Get color for this track ID
#         color = (colors[track_id % len(colors)] * 255).astype(np.uint8).tolist()
        
#         # Create text label
#         text = f'{track_id} : {score * 100:.1f}% | {len_feats}'
#         txt_color = (0, 0, 0) if np.mean(colors[track_id % len(colors)]) > 0.5 else (255, 255, 255)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         txt_size = cv2.getTextSize(text, font, 0.4 * m, 1 * m)[0]
        
#         # Draw bounding box
#         cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        
#         # Draw center point
#         cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
        
#         # Draw text background
#         txt_bk_color = (colors[track_id % len(colors)] * 255 * 0.7).astype(np.uint8).tolist()
#         cv2.rectangle(
#             img,
#             (x0, y0 - 1),
#             (x0 + txt_size[0] + 1, y0 - int(1.5 * txt_size[1])),
#             txt_bk_color,
#             -1
#         )
        
#         # Draw text
#         cv2.putText(img, text, (x0, y0 - txt_size[1]), font, 0.4 * m, txt_color, thickness=1 * m)
        
#     return img

def visualize(dets, img, colors, pose, pose_result, cur_frame):
    m = 2
    if len(dets) == 0:
        return img

    keypoints = [p['keypoints'][:,:2] for p in pose_result]
    scores = [p['keypoints'][:,2] for p in pose_result]
    img = visualize_kpt(img, keypoints, scores, thr=0.3)
            
    for obj in dets:
        score = obj[4]
        track_id = int(obj[5])
        len_feats = ' ' if obj[6] == 50 else obj[6]
        x0, y0, x1, y1 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])

        # Calculate center point of bounding box
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2

        color = (colors[track_id%80] * 255).astype(np.uint8).tolist()
        text = '{} : {:.1f}% | {}'.format(track_id, score * 100, len_feats)
        txt_color = (0, 0, 0) if np.mean(colors[track_id%80]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4*m, 1*m)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        # Draw circle at center of bounding box
        cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)  # Green circle at center

        txt_bk_color = (colors[track_id%80] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 - 1),
            (x0 + txt_size[0] + 1, y0 - int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 - txt_size[1]), font, 0.4*m, txt_color, thickness=1*m)
    
    return img

def visualize_kpt(img,
              keypoints,
              scores,
              thr=0.3) -> np.ndarray:

    skeleton = [
        [12, 13], [13, 0], [13, 1], [0, 1], [6, 7], [0, 2], [2, 4], 
        [1, 3], [3, 5], [0, 6], [1, 7], [6, 8], [8, 10], [7, 9], [9, 11]
    ]
    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]
    link_color = [3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
    point_color = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3]

    # draw keypoints and skeleton
    for kpts, score in zip(keypoints, scores):
        for kpt, color in zip(kpts, point_color):
            cv2.circle(img, tuple(kpt.astype(np.int32)), 2, palette[color], 2,
                       cv2.LINE_AA)
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                cv2.line(img, tuple(kpts[u].astype(np.int32)),
                         tuple(kpts[v].astype(np.int32)), palette[color], 1,
                         cv2.LINE_AA)
                
    return img