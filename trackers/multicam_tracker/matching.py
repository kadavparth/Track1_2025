import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from . import kalman_filter
import pdb

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def tlbr_expand(tlbr, scale=1.2):
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(atracks, btracks, metric='cosine'):
    """
    :param atracks: list[Appearance Features]
    :param btracks: list[Appearance Features]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    
    # Convert to proper numpy arrays
    try:
        atrack_features = np.asarray([feat for feat in atracks if feat is not None], dtype=float)
        btrack_features = np.asarray([feat for feat in btracks if feat is not None], dtype=float)
        
        # Check if we have valid features
        if len(atrack_features) == 0 or len(btrack_features) == 0:
            return np.zeros((len(atracks), len(btracks)), dtype=float)
            
        # Ensure 2D array
        if atrack_features.ndim == 1:
            atrack_features = atrack_features.reshape(1, -1)
        if btrack_features.ndim == 1:
            btrack_features = btrack_features.reshape(1, -1)
            
        cost_matrix = np.maximum(0.0, cdist(atrack_features, btrack_features, metric))
    except Exception as e:
        print(f"Warning: Error in embedding_distance: {e}")
        cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=float)
    
    return cost_matrix

def euclidean_distance(atracks, btracks, metric='euclidean'):
    """
    :param atracks: list[Locations]
    :param btracks: list[Locations]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    
    try:
        # Handle different input formats and ensure 2D array
        atrack_locs = []
        for loc in atracks:
            if isinstance(loc, (list, tuple)):
                if len(loc) >= 2:
                    atrack_locs.append([float(loc[0]), float(loc[1])])
                else:
                    atrack_locs.append([0.0, 0.0])  # Default location
            elif isinstance(loc, np.ndarray):
                if loc.size >= 2:
                    atrack_locs.append([float(loc.flat[0]), float(loc.flat[1])])
                else:
                    atrack_locs.append([0.0, 0.0])
            else:
                atrack_locs.append([0.0, 0.0])  # Default for invalid location
        
        btrack_locs = []
        for loc in btracks:
            if isinstance(loc, (list, tuple)):
                if len(loc) >= 2:
                    btrack_locs.append([float(loc[0]), float(loc[1])])
                else:
                    btrack_locs.append([0.0, 0.0])
            elif isinstance(loc, np.ndarray):
                if loc.size >= 2:
                    btrack_locs.append([float(loc.flat[0]), float(loc.flat[1])])
                else:
                    btrack_locs.append([0.0, 0.0])
            else:
                btrack_locs.append([0.0, 0.0])
        
        atrack_locs = np.asarray(atrack_locs, dtype=float)
        btrack_locs = np.asarray(btrack_locs, dtype=float)
        
        # Ensure we have valid 2D arrays
        if len(atrack_locs) == 0 or len(btrack_locs) == 0:
            return cost_matrix
            
        if atrack_locs.ndim != 2 or btrack_locs.ndim != 2:
            print(f"Warning: Invalid array dimensions in euclidean_distance: {atrack_locs.shape}, {btrack_locs.shape}")
            return cost_matrix
            
        if atrack_locs.shape[1] == 0 or btrack_locs.shape[1] == 0:
            print(f"Warning: Empty location arrays in euclidean_distance")
            return cost_matrix

        cost_matrix = cdist(atrack_locs, btrack_locs, metric)
        
    except Exception as e:
        print(f"Warning: Error in euclidean_distance: {e}")
        print(f"  atracks length: {len(atracks)}, btracks length: {len(btracks)}")
        if len(atracks) > 0:
            print(f"  First atrack type: {type(atracks[0])}, value: {atracks[0]}")
        if len(btracks) > 0:
            print(f"  First btrack type: {type(btracks[0])}, value: {btracks[0]}")
        cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=float)
    
    return cost_matrix

def centroid_distance(atracks, btracks):
    """
    Compute centroid distance between tracks
    :param atracks: list[STrack] or list of centroids
    :param btracks: list[STrack] or list of centroids
    :return: cost_matrix np.ndarray
    """
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=float)
    
    # Extract centroids from tracks
    if hasattr(atracks[0], 'centroid'):
        a_centroids = [track.centroid for track in atracks]
    else:
        a_centroids = atracks
        
    if hasattr(btracks[0], 'centroid'):
        b_centroids = [track.centroid for track in btracks]
    else:
        b_centroids = btracks
    
    return euclidean_distance(a_centroids, b_centroids)

def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost