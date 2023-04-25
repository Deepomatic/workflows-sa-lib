from PIL import Image
from lib.prediction_processing import denormalize_bbox
from lib.logging_config import logger


def get_crop(image, bbox, extension_factor=None, angle=0, resize_w=None, patchwork=False):
    "Takes a pil image"
    w = image.size[0]
    h = image.size[1]

    if bbox is not None:
        # for tag/cla, all coordinates are 0.0
        if bbox['xmax'] == 0.0:
            bbox['xmax'] = 1.0
        if bbox['ymax'] == 0.0:
            bbox['ymax'] = 1.0

        absolute_bbox = denormalize_bbox(bbox, w, h)
        crop_width = absolute_bbox['xmax'] - absolute_bbox['xmin']
        crop_height = absolute_bbox['ymax'] - absolute_bbox['ymin']
        if extension_factor is None:
            extension_factor = 0
        crop_coord = (
            absolute_bbox['xmin'] - crop_width * extension_factor,
            absolute_bbox['ymin'] - crop_height * extension_factor,
            absolute_bbox['xmax'] + crop_width * extension_factor,
            absolute_bbox['ymax'] + crop_height * extension_factor
        )
        image = image.crop(crop_coord)

    if angle != 0:
        image = image.rotate(- angle, expand="True")

    if resize_w is not None and (resize_w > crop_width):
        logger.debug(f'resizing {crop_width}')
        wpercent = resize_w / crop_width
        logger.debug((wpercent, resize_w, w))
        hsize = int(crop_height*wpercent)
        logger.debug((resize_w, hsize))
        image = image.resize((resize_w, hsize), Image.ANTIALIAS)

    if patchwork:
        new_im = Image.new('RGB', (crop_width * 3, crop_height * 2))
        new_im.paste(image, (0, 0))
        new_im.paste(image, (0, crop_height))
        bigger_image = image.resize((crop_width * 2, crop_height * 2), Image.ANTIALIAS)
        new_im.paste(bigger_image, (crop_width, 0))
        # new_im.save(f"./test.png", format="PNG")
        image = new_im

    # Uncomment below to save image locally
    # image.save(f"./{image.size[0]}.png", format="PNG")

    return image
