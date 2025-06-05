from pathlib import Path


# calibration_position = {
#     # Val
#     "scene_000": sorted([str(p) for p in Path("/workspace/videos/val/scene_000").glob("**/calibration.json")]),
#     # Test
#     "scene_000": sorted([str(p) for p in Path("/workspace/videos/test/scene_000").glob("**/calibration.json")])}

def get_calibration_position(scene_name):
    base_path = Path(f"data/{scene_name}")
    return sorted(str(p) for p in base_path.glob("**/calibration.json"))


calibration_position = {
    'Warehouse_002': [
        # 'data/Warehouse_002/calibration.json',
        'data/Warehouse_002/calibration.json'
    ]
}