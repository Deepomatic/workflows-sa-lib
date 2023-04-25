import copy

from deepomatic.workflows.v2.core import Region, Bbox


def create_region_from_prediction(preds, crop_reference_bbox=None):
    '''
    Will create a region in the format that Studio can plot
    Can also change the reference bbox used in all the predictions coordinates (by default it is the full image).
    (Useful for detection in detection, where you want the coordinates to be those of the full image)
    Args:
        preds (list or dict): List of predictions or single prediction, in the processed dict format.
        crop_reference_bbox (dict): Bbox of the crop that is the actual reference of the bbox in preds.
    '''
    if not isinstance(preds, list):
        preds = [preds]
    regions = []
    if not preds:
        return regions
    preds_temp = copy.deepcopy(preds)
    for pred in preds_temp:
        if 'bbox' not in pred.keys():
            continue

        if crop_reference_bbox is not None:
            bbox = pred['bbox']
            l, h = crop_reference_bbox['xmax'] - \
                crop_reference_bbox['xmin'], crop_reference_bbox['ymax'] - crop_reference_bbox['ymin']
            bbox['xmin'] = crop_reference_bbox['xmin'] + bbox['xmin'] * l
            bbox['xmax'] = crop_reference_bbox['xmax'] - (1 - bbox['xmax']) * l
            bbox['ymin'] = crop_reference_bbox['ymin'] + bbox['ymin'] * h
            bbox['ymax'] = crop_reference_bbox['ymax'] - (1 - bbox['ymax']) * h
            pred['bbox'] = bbox

        region = Region(
            entry_name='image',
            bbox=Bbox(xmin=pred['bbox']['xmin'], ymin=pred['bbox']['ymin'],
                      xmax=pred['bbox']['xmax'], ymax=pred['bbox']['ymax']),
            label_name=pred.get('concept', 'no_concept'),
            score=pred.get('score', 0.0)
        )
        regions.append(region)
    return regions
