import os
import tifffile as tiff
from os.path import basename, join
import numpy as np
import logging

logger = logging.getLogger(__name__)


def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def imsave(img, output, path, dtype=np.float32):
    """
    save a single image or multiple images.
    len(path) == img.shape[2]
    """
    if isinstance(path, list) or isinstance(path, tuple):
        for num, p in enumerate(path):
            filename = join(output, basename(p).split('.')[0]+'.tif')
            logger.debug('Image (shape {0}) is saved at {1}'.format(img.shape, filename))
            tiff.imsave(filename, img[:, :, num].astype(dtype))
    else:
        filename = join(output, basename(path).split('.')[0]+'.tif')
        logger.debug('Image (shape {0}) is saved at {1}'.format(img.shape, filename))
        tiff.imsave(filename, img.astype(dtype))
