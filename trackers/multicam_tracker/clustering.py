from . import matching
from itertools import combinations
import numpy as np
import pdb


class ID_Distributor:
    def __init__(self, init_id=0):
        self.cur_id = init_id

    def assign_id(self):
        self.cur_id += 1
        return self.cur_id

class Clustering:
    def __init__(self, appearance_thresh=0.25, euc_thresh=0.2, match_thresh=0.8, map_size=None):
        self.appearance_thresh = appearance_thresh
        self.match_thresh = match_thresh
        if map_size:
            self.euc_thresh = euc_thresh
        self.max_len = 1

    def update(self, trackers, cur_frame, scene):
        matched_ids = []
        tracker_pairs = list(combinations(trackers, 2))

        # First check for scene_000 and scene_001
        # if scene == 'scene_000' or scene == 'scene_001':
        #     hw_thresh, has_points_thresh, has_heads_thresh, num_kpts_thresh = 1.5, 8, 2, 18
        #     emb_thresh = 0.30
        #     euc_thresh = 1
        # elif int(scene.split('_')[1]) in range(61, 71):
        #     hw_thresh, has_points_thresh, has_heads_thresh, num_kpts_thresh = 1.5, 8, 2, 18
        #     emb_thresh = 0.30
        #     euc_thresh = 1
        # elif int(scene.split('_')[1]) in range(71, 81):
        #     hw_thresh, has_points_thresh, has_heads_thresh, num_kpts_thresh = 1.5, 8, 2, 18
        #     emb_thresh = 0.325
        #     euc_thresh = 1
        # elif int(scene.split('_')[1]) in range(81, 91):
        #     hw_thresh, has_points_thresh, has_heads_thresh, num_kpts_thresh = 1.5, 8, 2, 18
        #     emb_thresh = 0.30
        #     euc_thresh = 1.5
        # else:
        #     print('Not Test Set Scene')
        #     raise ValueError('Invalid scene number')
        hw_thresh, has_points_thresh, has_heads_thresh, num_kpts_thresh = 1.5, 8, 2, 18
        emb_thresh = 0.30
        euc_thresh = 1
        try:
            scene_number = int(scene.split('_')[1])
            if scene_number in range(61, 71):
                hw_thresh, has_points_thresh, has_heads_thresh, num_kpts_thresh = 1.5, 8, 2, 18
                emb_thresh = 0.30
                euc_thresh = 1
            elif scene_number in range(71, 81):
                hw_thresh, has_points_thresh, has_heads_thresh, num_kpts_thresh = 1.5, 8, 2, 18
                emb_thresh = 0.325
                euc_thresh = 1
            elif scene_number in range(81, 91):
                hw_thresh, has_points_thresh, has_heads_thresh, num_kpts_thresh = 1.5, 8, 2, 18
                emb_thresh = 0.30
                euc_thresh = 1.5
            else:
                raise ValueError
        except:
            print(f"[Warning] Scene '{scene}' not in predefined test set ranges. Using default clustering thresholds.")

        for tracker in trackers:
            for track in tracker.tracked_stracks:
                track.matched_dist = None

        for a_tracker, b_tracker in tracker_pairs:
            a_features, b_features, a_locations, b_locations, a_pose, b_pose = [], [], [], [], [], []
            a_match_to_id = {}
            num = 0
            for i, track in enumerate(a_tracker.tracked_stracks):
                if track.pose is not None:
                    has_heads, has_points, has_parts = self.pose_check_all(track.pose)
                else:
                    has_heads, has_points, has_parts = None, None, None
                t, l, w, h = track.tlwh.tolist()
                if h/w < hw_thresh or has_points is None or sum(has_points) <= has_points_thresh or sum(has_heads) <= has_heads_thresh or track.num_kpts >= num_kpts_thresh:
                    track.global_id = -1
                    continue

                a_features.append(track.curr_feat)
                a_locations.append(track.location[0])
                if track.pose is None:
                    a_pose.append([False] * 5)
                else:
                    a_pose.append(self.pose_check(track.pose))
                a_match_to_id[num] = i
                num += 1
            b_match_to_id = {}
            num = 0
            for i, track in enumerate(b_tracker.tracked_stracks):
                if track.pose is not None:
                    has_heads, has_points, has_parts = self.pose_check_all(track.pose)
                else:
                    has_heads, has_points, has_parts = None, None, None
                t, l, w, h = track.tlwh.tolist()
                if h/w < hw_thresh or has_points is None or sum(has_points) <= has_points_thresh or sum(has_heads) <= has_heads_thresh or track.num_kpts >= num_kpts_thresh:
                    track.global_id = -1
                    continue

                b_features.append(track.curr_feat)
                b_locations.append(track.location[0])
                if track.pose is None:
                    b_pose.append([False] * 5)
                else:
                    b_pose.append(self.pose_check(track.pose))
                b_match_to_id[num] = i
                num += 1

            euc_dists = matching.euclidean_distance(a_locations, b_locations) / self.max_len
            emb_dists = matching.embedding_distance(a_features, b_features) / 2

            if 0 not in emb_dists.shape:
                norm_emb_dists = (emb_dists - np.min(emb_dists)) / (np.max(emb_dists) - np.min(emb_dists))
                norm_euc_dists = (euc_dists - np.min(euc_dists)) / (np.max(euc_dists) - np.min(euc_dists))

                dists = np.zeros_like(euc_dists)
                for i in range(len(dists)):
                    for j in range(len(dists[0])):
                        ratio = sum((np.array(a_pose[i]) * np.array(b_pose[j]))) / 10

                        dists[i][j] += (1 - ratio) * norm_euc_dists[i][j] + ratio * norm_emb_dists[i][j]
            
            else:
                dists = 0.5 * euc_dists + 0.5 * emb_dists

            dists[emb_dists > emb_thresh] = 1.0
            dists[euc_dists > euc_thresh] = 1.0

            matches, u_afeats, u_bfeats = matching.linear_assignment(dists, thresh=0.999)

            for id_atrack, id_btrack in matches:
                a_global_id = a_tracker.tracked_stracks[a_match_to_id[id_atrack]].t_global_id
                b_global_id = b_tracker.tracked_stracks[b_match_to_id[id_btrack]].t_global_id
                matched_id = a_global_id if a_global_id <= b_global_id else b_global_id
                total_global_ids = [track.t_global_id for track in a_tracker.tracked_stracks] + [track.t_global_id for track in b_tracker.tracked_stracks]
                total_global_ids_lost = [track.t_global_id for track in a_tracker.lost_stracks] + [track.t_global_id for track in b_tracker.lost_stracks]
                if total_global_ids.count(matched_id) > 1 or total_global_ids_lost.count(matched_id):
                    continue

                if a_tracker.tracked_stracks[a_match_to_id[id_atrack]].matched_dist and a_global_id > b_global_id:
                    cur_dist = euc_dists[id_atrack][id_btrack] * emb_dists[id_atrack][id_btrack]
                    # cur_dist = dists[id_atrack][id_btrack]
                    old_dist = a_tracker.tracked_stracks[a_match_to_id[id_atrack]].matched_dist
                    if cur_dist > old_dist:
                        continue
                elif b_tracker.tracked_stracks[b_match_to_id[id_btrack]].matched_dist and a_global_id < b_global_id:
                    cur_dist = euc_dists[id_atrack][id_btrack] * emb_dists[id_atrack][id_btrack]
                    # cur_dist = dists[id_atrack][id_btrack]
                    old_dist = b_tracker.tracked_stracks[b_match_to_id[id_btrack]].matched_dist
                    if cur_dist > old_dist:
                        continue
                
                a_tracker.tracked_stracks[a_match_to_id[id_atrack]].t_global_id = b_tracker.tracked_stracks[b_match_to_id[id_btrack]].t_global_id = matched_id

                dist = euc_dists[id_atrack][id_btrack] * emb_dists[id_atrack][id_btrack]
                a_tracker.tracked_stracks[a_match_to_id[id_atrack]].matched_dist = dist
                b_tracker.tracked_stracks[b_match_to_id[id_btrack]].matched_dist = dist

        total_objects = [[track.t_global_id, track.curr_feat, track.location[0], track.pose, track.img_path, track.tlbr] for tracker in trackers for track in tracker.tracked_stracks if track.global_id != -1]        
        # total_objects = [[track.t_global_id, track.curr_feat, track.location[0], track.pose] for tracker in trackers for track in tracker.tracked_stracks if track.global_id != -1]
        total_ids = sorted(set([obj[0] for obj in total_objects]))
        groups = []
        for global_id in total_ids:
            num = 0
            features = []
            centroid = 0.0
            coords = []
            poses = []

            paths = []
            tlbrs = []
            for obj in total_objects:
                if obj[0] == global_id:
                    num += 1
                    features.append(obj[1])
                    centroid += obj[2]
                    coords.append(obj[2])
                    poses.append(obj[3])

                    paths.append(obj[4])
                    tlbrs.append(obj[5])
            centroid /= num
            groups.append([global_id, features, centroid, poses, paths, tlbrs, coords])
            # groups.append([global_id, features, centroid, poses])
        
        return np.array(groups, dtype=object)

    def pose_check(self, pose_):
        pose = pose_['keypoints']
        shoulder = pose[:2][:, 2].mean()
        hip = pose[[6, 7]][:, 2].mean()
        knee = pose[[8, 9]][:, 2].mean()
        head = pose[12][2]
        neck = pose[13][2]
        return np.array([head, neck, shoulder, hip, knee]) > 0.3
    
    def pose_check_all(self, pose_):
        pose = pose_['keypoints']

        left_shoulder = pose[0][2]
        right_shoulder = pose[1][2]
        shoulder = max(left_shoulder, right_shoulder)

        left_elbow = pose[2][2]
        right_elbow = pose[3][2]
        elbow = max(left_elbow, right_elbow)

        left_wrist = pose[4][2]
        right_wrist = pose[5][2]
        wrist = max(left_wrist, right_wrist)

        left_hip = pose[6][2]
        right_hip = pose[7][2]
        hip = max(left_hip, right_hip)

        left_knee = pose[8][2]
        right_knee = pose[9][2]
        knee = max(left_knee, right_knee)

        left_ankle = pose[10][2]
        right_ankle = pose[11][2]
        ankle = max(left_ankle, right_ankle)
        
        head = pose[12][2]
        neck = pose[13][2]

        return np.array([head, neck, left_shoulder, right_shoulder]) > 0.5, np.array([left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, 
                        left_knee, right_knee, left_ankle, right_ankle]) > 0.5, np.array([shoulder, elbow, wrist, hip, knee, ankle]) > 0.5

    def update_using_mctracker(self, trackers, mc_tracker):
        # mtrack_pools: [tracking_cluster_track1, tracking_cluster_track2, ...]
        mtrack_pools = []
        for track in mc_tracker.tracked_mtracks:
            if track.is_activated:
                mtrack_pools.append(track)

        # lost_mtracks: [lost_cluster_track1, lost_cluster_track2, ...]
        for track in mc_tracker.lost_mtracks:
            if mc_tracker.frame_id - track.end_frame <= 15:
                mtrack_pools.append(track)

        for tracker in trackers:
            """ First association with activated mtracks using embed distance"""
            tracks = [track for track in tracker.tracked_stracks]
            high_pose_tracks = []
            low_pose_tracks = []
            for track in tracks:
                if track.pose is not None and sum(self.pose_check(track.pose)) >= 4:
                    high_pose_tracks.append(track)
                else:
                    low_pose_tracks.append(track)

            sct_features = [track.curr_feat for track in high_pose_tracks]
            sct_locations = [track.location[0] for track in high_pose_tracks]

            mct_features = [feat for track in mtrack_pools for feat in list(track.features)]
            length_mcts = [len(track.features) for track in mtrack_pools]
            mct_centroids = [track.centroid for track in mtrack_pools]

            shape = (len(sct_features), len(length_mcts))
            if 0 in shape:
                dists = np.empty(shape)
            else:
                emb_dists = matching.embedding_distance(sct_features, mct_features) / 2.0
                emb_dists = group_dists(emb_dists, [1 for _ in range(len(sct_features))], length_mcts, shape, normalize=False)
                euc_dists = matching.euclidean_distance(sct_locations, mct_centroids) / self.max_len

                norm_emb_dists = (emb_dists - np.min(emb_dists)) / (np.max(emb_dists) - np.min(emb_dists))
                norm_euc_dists = (euc_dists - np.min(euc_dists)) / (np.max(euc_dists) - np.min(euc_dists))

                dists = 0.5 * norm_euc_dists + 0.5 * norm_emb_dists

            matches, u_scts, u_mcts = matching.linear_assignment(dists, thresh=0.999)
            for isct, imct in matches:
                high_pose_tracks[isct].global_id = mtrack_pools[imct].track_id

            """ Second association with lost mtracks using euclidean distance"""
            left_tracks = low_pose_tracks + [high_pose_tracks[i] for i in u_scts]
            left_mtracks = [mtrack_pools[i] for i in u_mcts]

            sct_locations = [track.location[0] for track in left_tracks]
            mct_centroids = [track.centroid for track in left_mtracks]

            euc_dists = matching.euclidean_distance(sct_locations, mct_centroids) / self.max_len
            dists = euc_dists

            matches, u_scts, u_mcts = matching.linear_assignment(dists, thresh=2)
            for isct, imct in matches:
                left_tracks[isct].global_id = left_mtracks[imct].track_id
            for i in u_scts:
                left_tracks[i].global_id = -2


def group_dists(rerank_dists, lengths_exists, lengths_new, shape, normalize=True):
    emb_dists = np.zeros(shape, dtype=float)
    total_sum = np.sum(rerank_dists)
    num = 0
    ratio = 0
    for i, len_e in enumerate(lengths_exists):
        for j, len_n in enumerate(lengths_new):
            start_x = sum(lengths_exists[:i])
            end_x = start_x + lengths_exists[i]
            start_y = sum(lengths_new[:j])
            end_y = start_y + lengths_new[j]
            # emb_dists[i,j] = np.sum(rerank_dists[start_x:end_x, start_y:end_y]) / total_sum
            if shape == (1,1):
                emb_dists[i,j] = np.mean(rerank_dists[start_x:end_x, start_y:end_y]) 
            else:
                emb_dists[i,j] = np.sum(rerank_dists[start_x:end_x, start_y:end_y]) / total_sum / (len_e*len_n)
            # emb_dists[i,j] = np.sum(rerank_dists[start_x:end_x, start_y:end_y]) 
    max_val = np.max(emb_dists)
    min_val = np.min(emb_dists)
    if shape != (1, 1) and normalize:
        emb_dists = (emb_dists - min_val) / (max_val - min_val)
    return emb_dists

