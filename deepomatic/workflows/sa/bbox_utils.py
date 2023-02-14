import numpy as np


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


def bb_intersection_over_union(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = [box1['xmin'], box1['ymin'], box1['xmax'], box1['ymax']]
    xmin2, ymin2, xmax2, ymax2 = [box2['xmin'], box2['ymin'], box2['xmax'], box2['ymax']]
    # Calculate the area of each rectangle
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1 The area of
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2 The area of
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


def bb_distance(boxA, boxB):
    # determine the distance between the center of the two bboxes
    centerA = ((boxA['xmin']+boxA['xmax'])/2, (boxA['ymin']+boxA['ymax'])/2)
    centerB = ((boxB['xmin']+boxB['xmax'])/2, (boxB['ymin']+boxB['ymax'])/2)
    # compute the distance betwen those two centers
    distance = np.sqrt((centerA[0]-centerB[0])**2+(centerA[1]-centerB[1])**2)
    return distance


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
    elif metric == 'overlap':
        ordered_preds = sorted(preds, key=lambda d: compute_sum_iou(d, preds))
    return ordered_preds
