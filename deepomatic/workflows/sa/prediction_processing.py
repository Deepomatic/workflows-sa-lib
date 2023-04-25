import numpy as np
import math


def bb_intersection_over_union(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = [box1['xmin'], box1['ymin'], box1['xmax'], box1['ymax']]
    xmin2, ymin2, xmax2, ymax2 = [box2['xmin'], box2['ymin'], box2['xmax'], box2['ymax']]
    # Calculate the area of each rectangle
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # The area of b1
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # The area of b2
    # Calculating intersecting rectangles
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G The area of
    a2 = s1 + s2 - a1
    if a2 == 0:
        # ERROR
        return 0
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou


def bb_intersection_over_area(box1, box2):
    """
    Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, IOA(box1, box2) != IOA(box2, box1).
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = [box1['xmin'], box1['ymin'], box1['xmax'], box1['ymax']]
    xmin2, ymin2, xmax2, ymax2 = [box2['xmin'], box2['ymin'], box2['xmax'], box2['ymax']]
    # Calculate the area of each rectangle
    # s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1 The area of
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2 The area of
    # Calculating intersecting rectangles
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G The area of
    if s2 == 0:
        # ERROR
        return 0
    ioa = a1 / s2  # iou = a1/ (s1 + s2 - a1)
    return ioa


def bb_center(box):
    """Get the center of the bbox"""
    center = ((box['xmin']+box['xmax'])/2, (box['ymin']+box['ymax'])/2)
    return center


def bb_distance(boxA, boxB, axis=None):
    """Get the distance between the center of the two bboxes on a given axis
    The x axis is the horizontal axis, the y axis is the vertical axis.
    If you don't specify any axis, the distance between the two centers is returned but using the relative coordinates.
    In theory, it doesn't make any sense, in pratice, it can still be usefu.
    """
    if axis is None:
        # determine the distance between the center of the two bboxes
        centerA = ((boxA['xmin'] + boxA['xmax']) / 2, (boxA['ymin'] + boxA['ymax']) / 2)
        centerB = ((boxB['xmin'] + boxB['xmax']) / 2, (boxB['ymin'] + boxB['ymax']) / 2)
        # compute the distance betwen those two centers
        distance = np.sqrt((centerA[0] - centerB[0]) ** 2 + (centerA[1] - centerB[1]) ** 2)
    elif axis == 'x':
        centerA = (boxA['xmin']+boxA['xmax'])/2
        centerB = (boxB['xmin']+boxB['xmax'])/2
        distance = abs(centerA - centerB)
    elif axis == 'y':
        centerA = (boxA['ymin']+boxA['ymax'])/2
        centerB = (boxB['ymin']+boxB['ymax'])/2
        distance = abs(centerA - centerB)
    else:
        raise ValueError("axis must be None, 'x' or 'y'")
    return distance


def predictions_reorder(preds, metric='xmin'):
    """
    Used to reorder a list predictions using a metric (from minimum to maximum)
    If metric = 'xmin', it will reorder them from left to right, just using the xmin metric
    If metric = 'overlap', it will reorder them from ones that less overlap each other to the ones that most.
    """
    def compute_sum_iou(pred, preds):
        somme_iou = 0
        for pred2 in preds:
            if pred == preds:
                continue
            somme_iou += bb_intersection_over_union(pred['bbox'], pred2['bbox'])
        return somme_iou
    if metric == 'xmin':
        ordered_preds = sorted(preds, key=lambda d: d['bbox']['xmin'])
    elif metric == 'ymin':
        ordered_preds = sorted(preds, key=lambda d: d['bbox']['ymin'])
    elif metric == 'overlap':
        ordered_preds = sorted(preds, key=lambda d: compute_sum_iou(d, preds))
    return ordered_preds


def predictions_filter_recursively(preds, metric='iou', threshold=0.5, kept_preds=[]):
    """
    Used to filter recursively certain predictions from a list of predictions using a metric and a threshold.
    If metric = 'iou', it will filter the predictions and keep only the ones with the highest score among those having an iou above the given threshold.
    If metric = 'ioa', it will filter the predictions and keep only the ones with the highest score among those having an ioa above the given threshold.
    """
    # Make sure that all preds are ordered by their score
    preds = sorted(preds, key=lambda d: d['score'], reverse=True)
    # Exit condition
    if len(preds) == 0:
        return kept_preds
    elif len(preds) == 1:
        kept_preds.append(preds[0])
        preds.pop(0)
        return predictions_filter_recursively(preds, metric=metric, threshold=threshold, kept_preds=kept_preds)
    else:
        pred1 = preds[0]
        metric_list = []
        for idx in range(len(preds)):
            pred2 = preds[idx]
            if metric == 'ioa':
                metric_list.append(bb_intersection_over_area(pred1['bbox'], pred2['bbox']))
            elif metric == 'iou':
                metric_list.append(bb_intersection_over_union(pred1['bbox'], pred2['bbox']))
        # Get a list of booleans, saying if this pred is above threshold
        # bool_list = [(x > threshold) for x in metric_list]
        # We keep pred1 as we know it has a higher score than others (ordered list) and we pop the other elements
        kept_preds.append(preds[0])
        # We now need to remove predictions that had their metric above threshold
        idx_to_del = []
        for idx, metric_value in enumerate(metric_list):
            if metric_value > threshold:
                idx_to_del.append(idx)
        # We reverse the list in order to pop elements easily
        for i in sorted(idx_to_del, reverse=True):
            del preds[i]
        # Now we do the recursion to do this for the second element
        return predictions_filter_recursively(preds, metric=metric, threshold=threshold, kept_preds=kept_preds)


def denormalize_bbox(bbox, w, h):
    '''
    Will denormalize a bounding box coordinates. Ex: {xmin: 0.23, ymin: 0.78 ...} => {xmin: 156, ymin: 593 ...}
    '''
    bbox = {'xmin': int(bbox['xmin'] * w), 'ymin': int(bbox['ymin'] * h),
            'xmax': int(bbox['xmax'] * w), 'ymax': int(bbox['ymax'] * h)}
    return bbox


def normalize_bbox(bbox, w, h):
    '''
    Will normalize a bounding box coordinates. Ex: {xmin: 156, ymin: 593 ...} => {xmin: 0.23, ymin: 0.78 ...}
    '''
    bbox = {'xmin': float(bbox['xmin']/w), 'ymin': float(bbox['ymin']/h),
            'xmax': float(bbox['xmax']/w), 'ymax': float(bbox['ymax']/h)}
    return bbox


def bb_match(bbox_gt, bbox_pred, metric='iou', IOU_THRESH=0.5, IOA_THRESH=0.5, DISTANCE_THRESH=1, DISTANCE_AXIS=None):
    '''
    Used to match a set of bboxes to another one. Here there are called gt and pred, but it is just a naming.
    The metric used to do the matching can either be the iou or the distance between the boxes.
    Args:
    bbox_gt (list): The context object, which will be used to initialize the region object
    bbox_pred (list): List of predictions or single prediction, in the processed dict format.
    metric (string): 'iou', 'ioa', or 'distance'
    Returns:
    idxs_true (list): List of indexes in bbox_gt
    idxs_pred: List of matching indexes in bbox_pred
    '''

    # Some necessary functions
    def simple_optimize(matrix):
        '''
        Pour remplacer scipy.optimize.linear_sum_assignment en attendant de l'ajouter au workflow-server.
        '''
        # L'objectif est de venir renvoyer 2 sets d'indices.
        matrix_temp = np.copy(matrix)
        idxs_A = np.zeros(matrix_temp.shape[0], dtype=np.int64)
        idxs_B = np.zeros(matrix_temp.shape[0], dtype=np.int64)
        for i in range(matrix_temp.shape[0]):
            xmin, ymin = find_minimum_idx(matrix_temp)
            idxs_A[xmin] = xmin
            idxs_B[xmin] = ymin
            matrix_temp[:, ymin] = math.inf
            matrix_temp[xmin, :] = math.inf
        return idxs_A, idxs_B

    def find_minimum_idx(matrix):
        '''
        Find the X,Y coordinates of the minimum element in a numpy array.
        '''
        min_per_line = np.min(matrix, axis=1)
        global_min = np.min(min_per_line)
        idx_min_line = np.where(min_per_line == global_min)[0][0]
        idx_min_col = np.where(matrix[idx_min_line] == global_min)[0][0]

        return idx_min_line, idx_min_col

    n_true = len(bbox_gt)
    n_pred = len(bbox_pred)

    MAX_DIST = 1.0
    MIN_IOU = 0.0
    MIN_IOA = 0.0
    if metric == 'iou':
        default_metric = MIN_IOU
    elif metric == 'ioa':
        default_metric = MIN_IOA
    elif metric == 'distance':
        default_metric = MAX_DIST

    # NUM_GT x NUM_PRED
    metric_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            if metric == 'iou':
                metric_matrix[i, j] = bb_intersection_over_union(bbox_gt[i]['bbox'], bbox_pred[j]['bbox'])
            elif metric == 'ioa':
                metric_matrix[i, j] = bb_intersection_over_area(bbox_gt[i]['bbox'], bbox_pred[j]['bbox'])
            elif metric == 'distance':
                metric_matrix[i, j] = bb_distance(
                    bbox_gt[i]['bbox'], bbox_pred[j]['bbox'], axis=DISTANCE_AXIS)

    if n_pred > n_true:
        # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        metric_matrix = np.concatenate((metric_matrix, np.full((diff, n_pred), default_metric)), axis=0)
    if n_true > n_pred:
        # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        metric_matrix = np.concatenate((metric_matrix, np.full((n_true, diff), default_metric)), axis=1)

    # Call the Hungarian matching
    # TODO: Add scipy to workflow server to do a smart matching
    # For now we will do the matching more simply
    if metric == 'iou' or metric == 'ioa':
        idxs_true, idxs_pred = linear_sum_assignment(1-metric_matrix)
        # idxs_true, idxs_pred = simple_optimize(1-metric_matrix)
    elif metric == 'distance':
        idxs_true, idxs_pred = linear_sum_assignment(metric_matrix)
        # idxs_true, idxs_pred = simple_optimize(metric_matrix)
    # if (not idxs_true.size) or (not idxs_pred.size):
    #     metrics = np.array([])
    # else:
    #     metrics = metric_matrix[idxs_true, idxs_pred]
    # remove dummy assignments
    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    metrics_actual = metric_matrix[idx_gt_actual, idx_pred_actual]
    if metric == 'iou':
        sel_valid = (metrics_actual > IOU_THRESH)
    elif metric == 'ioa':
        sel_valid = (metrics_actual > IOA_THRESH)
    elif metric == 'distance':
        sel_valid = (metrics_actual < DISTANCE_THRESH)
    label = sel_valid.astype(int)
    # print(idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], metrics_actual[sel_valid], label)
    return list(idx_gt_actual[sel_valid]), list(idx_pred_actual[sel_valid]), list(metrics_actual[sel_valid]), list(label)


########### REPLACE BY ADDING SCIPY TO WORKFLOW SERVER ###########
# Taken from https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py#L13-L107
# Hungarian algorithm (Kuhn-Munkres) for solving the linear sum assignment
# problem.

def linear_sum_assignment(cost_matrix):
    """Solve the linear sum assignment problem.
    """
    cost_matrix = np.asarray(cost_matrix)
    if len(cost_matrix.shape) != 2:
        raise ValueError("expected a matrix (2-d array), got a %r array"
                         % (cost_matrix.shape,))

    # The algorithm expects more columns than rows in the cost matrix.
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    state = _Hungary(cost_matrix)

    # No need to bother with assignments if one of the dimensions
    # of the cost matrix is zero-length.
    step = None if 0 in cost_matrix.shape else _step1

    while step is not None:
        step = step(state)

    if transposed:
        marked = state.marked.T
    else:
        marked = state.marked
    return np.where(marked == 1)


class _Hungary(object):
    """State of the Hungarian algorithm.
    Parameters
    ----------
    cost_matrix : 2D matrix
        The cost matrix. Must have shape[1] >= shape[0].
    """

    def __init__(self, cost_matrix):
        self.C = cost_matrix.copy()

        n, m = self.C.shape
        self.row_uncovered = np.ones(n, dtype=bool)
        self.col_uncovered = np.ones(m, dtype=bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.zeros((n + m, 2), dtype=int)
        self.marked = np.zeros((n, m), dtype=int)

    def _clear_covers(self):
        """Clear all covered matrix cells"""
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True


# Individual steps of the algorithm follow, as a state machine: they return
# the next step to be taken (function to be called), if any.

def _step1(state):
    """Steps 1 and 2 in the Wikipedia page."""

    # Step 1: For each row of the matrix, find the smallest element and
    # subtract it from every element in its row.
    state.C -= state.C.min(axis=1)[:, np.newaxis]
    # Step 2: Find a zero (Z) in the resulting matrix. If there is no
    # starred zero in its row or column, star Z. Repeat for each element
    # in the matrix.
    for i, j in zip(*np.where(state.C == 0)):
        if state.col_uncovered[j] and state.row_uncovered[i]:
            state.marked[i, j] = 1
            state.col_uncovered[j] = False
            state.row_uncovered[i] = False

    state._clear_covers()
    return _step3


def _step3(state):
    """
    Cover each column containing a starred zero. If n columns are covered,
    the starred zeros describe a complete set of unique assignments.
    In this case, Go to DONE, otherwise, Go to Step 4.
    """
    marked = (state.marked == 1)
    state.col_uncovered[np.any(marked, axis=0)] = False

    if marked.sum() < state.C.shape[0]:
        return _step4


def _step4(state):
    """
    Find a noncovered zero and prime it. If there is no starred zero
    in the row containing this primed zero, Go to Step 5. Otherwise,
    cover this row and uncover the column containing the starred
    zero. Continue in this manner until there are no uncovered zeros
    left. Save the smallest uncovered value and Go to Step 6.
    """
    # We convert to int as numpy operations are faster on int
    C = (state.C == 0).astype(int)
    covered_C = C * state.row_uncovered[:, np.newaxis]
    covered_C *= np.asarray(state.col_uncovered, dtype=int)
    n = state.C.shape[0]
    m = state.C.shape[1]

    while True:
        # Find an uncovered zero
        row, col = np.unravel_index(np.argmax(covered_C), (n, m))
        if covered_C[row, col] == 0:
            return _step6
        else:
            state.marked[row, col] = 2
            # Find the first starred element in the row
            star_col = np.argmax(state.marked[row] == 1)
            if state.marked[row, star_col] != 1:
                # Could not find one
                state.Z0_r = row
                state.Z0_c = col
                return _step5
            else:
                col = star_col
                state.row_uncovered[row] = False
                state.col_uncovered[col] = True
                covered_C[:, col] = C[:, col] * (
                    np.asarray(state.row_uncovered, dtype=int))
                covered_C[row] = 0


def _step5(state):
    """
    Construct a series of alternating primed and starred zeros as follows.
    Let Z0 represent the uncovered primed zero found in Step 4.
    Let Z1 denote the starred zero in the column of Z0 (if any).
    Let Z2 denote the primed zero in the row of Z1 (there will always be one).
    Continue until the series terminates at a primed zero that has no starred
    zero in its column. Unstar each starred zero of the series, star each
    primed zero of the series, erase all primes and uncover every line in the
    matrix. Return to Step 3
    """
    count = 0
    path = state.path
    path[count, 0] = state.Z0_r
    path[count, 1] = state.Z0_c

    while True:
        # Find the first starred element in the col defined by
        # the path.
        row = np.argmax(state.marked[:, path[count, 1]] == 1)
        if state.marked[row, path[count, 1]] != 1:
            # Could not find one
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Find the first prime element in the row defined by the
        # first path step
        col = np.argmax(state.marked[path[count, 0]] == 2)
        if state.marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    # Convert paths
    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state.marked[path[i, 0], path[i, 1]] = 0
        else:
            state.marked[path[i, 0], path[i, 1]] = 1

    state._clear_covers()
    # Erase all prime markings
    state.marked[state.marked == 2] = 0
    return _step3


def _step6(state):
    """
    Add the value found in Step 4 to every element of each covered row,
    and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered lines.
    """
    # the smallest uncovered value in the matrix
    if np.any(state.row_uncovered) and np.any(state.col_uncovered):
        minval = np.min(state.C[state.row_uncovered], axis=0)
        minval = np.min(minval[state.col_uncovered])
        state.C[~state.row_uncovered] += minval
        state.C[:, state.col_uncovered] -= minval
    return _step4
