
import numpy as np
import cv2
import json
from deepomatic.workflows.v2.cv import google_ocr
from lib.bbox_utils import denormalize_bbox, normalize_bbox, bb_intersection_over_union, bb_intersection_over_area
from google.cloud.vision_v1 import AnnotateImageResponse


class ProcessedGoogleOCR():
    def __init__(self, image, image_context={"language_hints": ["en"]}, debug=False):

        self.debug = debug
        if debug:
            nparr = np.fromstring(image, np.uint8)
            self.img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Get height and width of image
        self.h, self.w = image.height, image.width

        # Run text detection
        self.raw_response = google_ocr(image, image_context=image_context)
        self.response_json = json.loads(AnnotateImageResponse.to_json(self.raw_response))
        if self.raw_response.text_annotations:
            self.text = self.raw_response.text_annotations[0].description
        else:
            self.text = ""

    def match_text_with_detection(self, bbox, default='', iou_thresh=0.5, ioa_thresh=0.5, mode='all'):
        """
        Set default to 'text' to return the text read even if no bbox is detected
        Set mode to 'best' or 'all':
            'best_word' if you just want to keep only the best 'word' in the ocr predictions,
            'sorted_words' if you want all the matched words, but ordered according to highest ioa (separated with spaces)
            'all' if you want all the matched words to be kept (separated with spaces)
        """
        # Comportement chelou, à voir si ça fait sens de garder
        if bbox is None:
            if default == 'text':
                return self.text
            return ""
        text = ""
        if self.debug:
            self._draw_bbox(denormalize_bbox(bbox, self.w, self.h), color=(0, 255, 0))
        annotation = self.response_json['textAnnotations']
        max_iou, max_ioa = 0, 0
        temp_info = []
        for word in annotation:
            bbox_google = self._get_google_bbox(word)
            iou = bb_intersection_over_union(
                denormalize_bbox(bbox, self.w, self.h), bbox_google)
            ioa = bb_intersection_over_area(
                denormalize_bbox(bbox, self.w, self.h), bbox_google)

            if iou > iou_thresh or ioa > ioa_thresh:
                # logger.debug(f"{word}, iou {iou}, ioa {ioa}")
                if self.debug:
                    self._draw_bbox(bbox_google)
                if mode == 'all':
                    text += word['description'] + ' '
                elif mode == 'best_word' and (iou > max_iou or ioa > max_ioa):
                    max_iou, max_ioa = iou, ioa
                    text = word['description']
                elif mode == 'sorted_words':
                    temp_info.append({'text': word['description'], 'ioa': ioa})

        # save the output image
        # cv2.imwrite('detect_text.png', self.img_np)

        if mode == 'sorted_words':
            sorted_temp_info = sorted(temp_info, key=lambda d: d['ioa'])
            sorted_string = ' '.join([x['text'] for x in sorted_temp_info])
            return sorted_string
        return text

    def match_with_detection(self, bbox, default='', iou_thresh=0.5, ioa_thresh=0.5, mode='all'):
        """
        Set default to 'text' to return the text read even if no bbox is detected
        Set mode to 'best' or 'all':
            'best_word' if you just want to keep only the best 'word' in the ocr predictions,
            'all' if you want all the matched words to be kept (separated with spaces)
        """

    def locate_word_in_text(self, searched_word):
        annotation = self.response_json['textAnnotations']
        for word in annotation:
            if word['description'] == searched_word:
                bbox_google = self._get_google_bbox(word)
                if self.debug:
                    self._draw_bbox(bbox_google)
                bbox_deepo = normalize_bbox(bbox_google, self.w, self.h)
                return bbox_deepo
        # save the output image
        # cv2.imwrite('/app/project/detect_text.png', self.img_np)
        return None

    def locate_sentence_in_text(self, searched_sentence):
        '''
        Args:
            searched_sentence: string to be found in text
        Returns:
            Returns bbox of sentence
        '''
        annotation = self.response_json['textAnnotations']
        temp_sentence = ""
        all_boxes = []
        for word in annotation:
            if word['description'] in searched_sentence:
                temp_sentence += word['description']
                all_boxes.append(self._get_google_bbox(word))
                # Removing spaces for comparison
                if temp_sentence == searched_sentence.replace(' ', ''):
                    # Merging boxes
                    merged_bbox = self._merge_google_bboxes(all_boxes)
                    if self.debug:
                        self._draw_bbox(merged_bbox)
                    bbox_deepo = normalize_bbox(merged_bbox, self.w, self.h)
                    return bbox_deepo
            else:
                temp_sentence = ""
                all_boxes = []

        # save the output image
        # cv2.imwrite('/app/project/detect_text.png', self.img_np)
        return None

    def _draw_bbox(self, bbox, color=(0, 0, 255), thickness=1):
        #
        self.img_np = cv2.rectangle(self.img_np, (bbox['xmin'], bbox['ymin']),
                                    (bbox['xmax'], bbox['ymax']), color, thickness)

    def _get_google_bbox(self, word):
        min_x = min(word['boundingPoly']['vertices'][0]['x'], word['boundingPoly']['vertices'][1]
                    ['x'], word['boundingPoly']['vertices'][2]['x'], word['boundingPoly']['vertices'][3]['x'])
        max_x = max(word['boundingPoly']['vertices'][0]['x'], word['boundingPoly']['vertices'][1]
                    ['x'], word['boundingPoly']['vertices'][2]['x'], word['boundingPoly']['vertices'][3]['x'])
        min_y = min(word['boundingPoly']['vertices'][0]['y'], word['boundingPoly']['vertices'][1]
                    ['y'], word['boundingPoly']['vertices'][2]['y'], word['boundingPoly']['vertices'][3]['y'])
        max_y = max(word['boundingPoly']['vertices'][0]['y'], word['boundingPoly']['vertices'][1]
                    ['y'], word['boundingPoly']['vertices'][2]['y'], word['boundingPoly']['vertices'][3]['y'])
        bbox = {'xmin': min_x, 'ymin': min_y, 'xmax': max_x, 'ymax': max_y}
        return bbox

    def _merge_google_bboxes(self, bboxes):
        new_bbox = {
            'xmin': min([x['xmin'] for x in bboxes]),
            'ymin': min([x['ymin'] for x in bboxes]),
            'xmax': max([x['xmax'] for x in bboxes]),
            'ymax': max([x['ymax'] for x in bboxes])}
        return new_bbox
