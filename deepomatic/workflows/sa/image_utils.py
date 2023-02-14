import io
from PIL import Image
from deepomatic.workflows.utils import load_image
from lib.bbox_utils import denormalize_bbox
from lib.logging_utils import logger


def get_crop(image_blob, bbox, extension_factor=None, angle=0, resize_w=None, patchwork=False):
    image = load_image(io.BytesIO(image_blob))
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

    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format="PNG")
    image_byte_array = image_byte_array.getvalue()

    # Uncomment below to save image locally
    # image.save(f"./{image.size[0]}.png", format="PNG")

    return image_byte_array
