from pathlib import Path


calibration_position = {
    # Val
    "scene_000": sorted([str(p) for p in Path("/workspace/videos/val/scene_000").glob("**/calibration.json")]),
    # Test
    "scene_000": sorted([str(p) for p in Path("/workspace/videos/test/scene_000").glob("**/calibration.json")])}
