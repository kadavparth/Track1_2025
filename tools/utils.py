import numpy as np
from pathlib import Path
import cv2
import os
import pickle
import json


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

sources = {
        # Val
        'scene_000': sorted([os.path.join('/workspace/frames/val/scene_000', p) for p in os.listdir('/workspace/frames/val/scene_000')]),
        # Test
        'scene_000': sorted([os.path.join('/workspace/frames/test/scene_000', p) for p in os.listdir('/workspace/frames/test/scene_000')])}

result_paths = {
    # Val
    'scene_000': './results/scene_000.txt',
    # Test
    'scene_000': './results/scene_000.txt'}

cam_ids = {
    # Val
    'scene_000': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_000') if cam.startswith('camera_')]),
    # Test
    'scene_000': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_000') if cam.startswith('camera_')])}

cam_ids_full = {
    # Val
    'scene_000': sorted([cam for cam in os.listdir('/workspace/videos/val/scene_000') if cam.startswith('camera_')]),
    # Test
    'scene_000': sorted([cam for cam in os.listdir('/workspace/videos/test/scene_000') if cam.startswith('camera_')])}

def get_reader_writer(source):
    src_paths = sorted(os.listdir(source),  key=lambda x: int(x.split("_")[-1].split(".")[0]))
    src_paths = [os.path.join(source, s) for s in src_paths]

    fps = 30
    wi, he = 1920, 1080
    os.makedirs('output_videos/' + source.split('/')[-2], exist_ok=True)
    # dst = 'output_videos/' + source.replace('/','').replace('.','') + '.mp4'
    dst = f"output_videos/{source.split('/')[-2]}/" + source.split('/')[-3] + '_' + source.split('/')[-2] + source.split('/')[-1] + '.mp4'
    video_writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (wi, he))
    print(f"{source}'s total frames: {len(src_paths)}")

    return [src_paths, video_writer]

def finalize_cams(src_handlers):
    for s, w in src_handlers:
        w.release()
        print(f"{w} released")

def write_vids(trackers, imgs, src_handlers, pose, colors, mc_tracker, cur_frame=0):
    writers = [w for s, w in src_handlers]
    gid_2_lenfeats = {}
    for track in mc_tracker.tracked_mtracks + mc_tracker.lost_mtracks:
        if track.is_activated:
            gid_2_lenfeats[track.track_id] = len(track.features)
        else:
            gid_2_lenfeats[-2] = len(track.features)

    for tracker, img, w in zip(trackers, imgs, writers):
        outputs = [t.tlbr.tolist() + [t.score, t.global_id, gid_2_lenfeats.get(t.global_id, -1)] for t in tracker.tracked_stracks]
        pose_result = [t.pose for t in tracker.tracked_stracks if t.pose is not None]
        img = visualize(outputs, img, colors, pose, pose_result, cur_frame)
        
        # Save just the first frame as an image
        if cur_frame < 100:  # or any other frame number you want to save
            cv2.imwrite('image1.jpg', img)
            print("Saved frame as image1.jpg")
        
        w.write(img)

        return outputs

def write_results_testset(result_lists, result_path):
    dst_folder = str(Path(result_path).parent)
    os.makedirs(dst_folder, exist_ok=True)
    # write multicam results
    with open(result_path, 'w') as f:
        print(result_path)
        for result in result_lists:
            for r in result:
                t, l, w, h = r['tlwh']
                xworld, yworld = r['2d_coord']
                row = [r['cam_id'], r['track_id'], r['frame_id'], int(t), int(l), int(w), int(h), float(xworld), float(yworld)]
                row = " ".join([str(r) for r in row]) + '\n'
                # row = " ".join(row)
                f.write(row)

def update_result_lists_testset(trackers, result_lists, frame_id, cam_ids, scene):
    results_frame = [[] for i in range(len(result_lists))]
    results_frame_feat = []
    for tracker, result_frame, result_list, cam_id in zip(trackers, results_frame, result_lists, cam_ids):
        for track in tracker.tracked_stracks:
            if track.global_id < 0: continue
            result = {
                'cam_id': int(cam_id),
                'frame_id': frame_id,
                'track_id': track.global_id,
                'sct_track_id': track.track_id,
                'tlwh': list(map(lambda x: int(x), track.tlwh.tolist())),
                '2d_coord': track.location[0].tolist()
            }
            result_ = list(result.values())
            result_list.append(result)

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