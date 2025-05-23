import os
import json
from pathlib import Path

# Define class mapping
CLASS_MAPPING = {
    "Forklift": 0,
    "NovaCarter": 1,
    "Person": 2,
    "Transporter": 3
}

def create_yolo_label(bbox, img_width=1920, img_height=1080):
    """
    Convert bounding box to YOLO format (x_center, y_center, width, height) normalized
    bbox format: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate center coordinates and dimensions
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Normalize coordinates
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]

def process_ground_truth(gt_path, output_dir):
    """
    Process ground truth file and create YOLO labels for Camera_0001
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read ground truth file
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    # Process each frame
    for frame_id, objects in gt_data.items():
        # Convert frame_id to integer (1-based indexing)
        frame_num = int(frame_id) + 1
        
        # Skip if frame number is outside our range (1-9000)
        if frame_num < 1 or frame_num > 9000:
            continue
            
        # List to store annotations for this frame
        frame_annotations = []
        
        # Process each object in the frame
        for obj in objects:
            obj_type = obj["object type"]
            if obj_type not in CLASS_MAPPING:
                print(f"Warning: Unknown object type {obj_type}, skipping...")
                continue
                
            class_id = CLASS_MAPPING[obj_type]
            
            # Get 2D bounding box for Camera_0001
            bboxes = obj["2d bounding box visible"]
            if "Camera_0001" not in bboxes:
                continue
                
            # Convert bbox to YOLO format
            yolo_bbox = create_yolo_label(bboxes["Camera_0001"])
            frame_annotations.append((class_id, yolo_bbox))
        
        # Create label file with name exactly matching the image filename
        label_file = os.path.join(output_dir, f"scene_000camera_0001_{frame_num}.txt")
        
        with open(label_file, 'w') as f:
            for class_id, yolo_bbox in frame_annotations:
                # Write to file: class_id x_center y_center width height
                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

def main():
    # Paths
    gt_file = "data/videos/train/scene_000/ground_truth.json"
    output_dir = "data/labels/train/scene_000"
    
    if not os.path.exists(gt_file):
        print(f"Ground truth file not found: {gt_file}")
        return
        
    print("Processing ground truth file for Camera_0001...")
    process_ground_truth(gt_file, output_dir)
    print("Finished processing ground truth")

if __name__ == "__main__":
    main()
