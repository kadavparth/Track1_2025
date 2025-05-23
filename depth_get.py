import h5py
import numpy as np
import cv2

# Read the H5 file and save depth map
with h5py.File('/home/pkd/Downloads/aicity/Camera_0001.h5', 'r') as f:
    # Get the specific dataset
    for key in f.keys():
        key = key
        depth_map = f[key][:]
        
        # Print detailed depth information
        print(f"Depth map statistics:")
        print(f"Shape: {depth_map.shape}")
        print(f"Data type: {depth_map.dtype}")
        print(f"Min value: {np.min(depth_map)}")
        print(f"Max value: {np.max(depth_map)}")
        print(f"Mean value: {np.mean(depth_map)}")
        print(f"Value at (461, 1657): {depth_map[656, 90]}")
        
        # Check if there are any metadata attributes that might indicate units
        print("\nDataset attributes:")
        for attr_name in f[key].attrs:
            print(f"{attr_name}: {f[key].attrs[attr_name]}")
        
        # Save to numpy array
        np.save('depth_map.npy', depth_map)
        print(f"Saved depth map to depth_map.npy")
        print(f"Array shape: {depth_map.shape}")
        print(f"Min value: {np.min(depth_map)}")
        print(f"Max value: {np.max(depth_map)}")
        
        # Visualize the depth map
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_uint8 = depth_map_normalized.astype(np.uint8)
        
        print(depth_map_uint8[1043, 90])

        cv2.imshow('Depth Map', depth_map_uint8)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break

# Later, you can load the depth maps like this:
# depth_maps = np.load('depth_maps.npy')
