import numpy as np
from pathlib import Path
import cv2
import os
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

# Adjusted for your Warehouse_002 dataset
# sources = {
#     'Warehouse_002': [
#         # sorted([os.path.join('data/Warehouse_002/frames/Camera_0001', p) for p in os.listdir('data/Warehouse_002/frames/Camera_0001')]),
#         sorted([os.path.join('data/Warehouse_002/frames/Camera_0003', p) for p in os.listdir('data/Warehouse_002/frames/Camera_0003')])
#     ]
# }

sources = {
    'Warehouse_002': [
        # 'data/Warehouse_002/frames/Camera_0001',
        # 'data/Warehouse_002/frames/Camera_0002',
        'data/Warehouse_002/frames/Camera_0003'
    ]
}

result_paths = {
    'Warehouse_002': './results/Warehouse_002.txt'
}

cam_ids = {
    'Warehouse_002': [1, 3]
}

cam_ids_full = {
    'Warehouse_002': ['Camera_0003']
    # 'Warehouse_002': ['Camera_0001', 'Camera_0003']
}

def get_reader_writer(source):
    src_paths = sorted(os.listdir(source), key=lambda x: int(x.split("_")[-1].split(".")[0].replace("frame", "")))
    src_paths = [os.path.join(source, s) for s in src_paths]
    fps = 30
    wi, he = 1920, 1080
    os.makedirs('output_videos/' + source.split('/')[-2], exist_ok=True)
    dst = f"output_videos/{source.split('/')[-2]}/{source.split('/')[-1]}.mp4"
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
        img = visualize(outputs, img, colors, cur_frame)
        w.write(img)
    return outputs

def write_results_testset(result_lists, result_path):
    dst_folder = str(Path(result_path).parent)
    os.makedirs(dst_folder, exist_ok=True)
    with open(result_path, 'w') as f:
        print(result_path)
        for result in result_lists:
            for r in result:
                t, l, w, h = r['tlwh']
                xworld, yworld = r['2d_coord']
                row = [r['cam_id'], r['track_id'], r['frame_id'], int(t), int(l), int(w), int(h), float(xworld), float(yworld)]
                f.write(" ".join([str(r) for r in row]) + '\n')

def update_result_lists_testset(trackers, result_lists, frame_id, cam_ids, scene):
    results_frame = [[] for _ in range(len(result_lists))]
    for tracker, result_frame, result_list, cam_id in zip(trackers, results_frame, result_lists, cam_ids):
        for track in tracker.tracked_stracks:
            if track.global_id < 0:
                continue
            result = {
                'cam_id': int(cam_id),
                'frame_id': frame_id,
                'track_id': track.global_id,
                'sct_track_id': track.track_id,
                'tlwh': list(map(int, track.tlwh.tolist())),
                '2d_coord': track.location[0].tolist()
            }
            result_list.append(result)

def visualize(dets, img, colors, cur_frame):
    m = 2
    for obj in dets:
        score = obj[4]
        track_id = int(obj[5])
        len_feats = obj[6]
        x0, y0, x1, y1 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        color = (colors[track_id % 80] * 255).astype(np.uint8).tolist()
        text = f'{track_id} : {score * 100:.1f}% | {len_feats}'
        txt_color = (0, 0, 0) if np.mean(colors[track_id % 80]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.4 * m, 1 * m)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
        txt_bk_color = (colors[track_id % 80] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 - 1),
            (x0 + txt_size[0] + 1, y0 - int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 - txt_size[1]), font, 0.4 * m, txt_color, thickness=1 * m)
    return img

