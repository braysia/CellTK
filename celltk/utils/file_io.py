import os
import tifffile as tiff
from os.path import basename, join
import numpy as np


def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def imsave(img, output, path):
    """
    save a single image or multiple images.
    len(path) == img.shape[2]
    """
    if isinstance(path, list) or isinstance(path, tuple):
        for num, p in enumerate(path):
            tiff.imsave(join(output, basename(p)), img[:, :, num].astype(np.float32))
    else:
        tiff.imsave(join(output, basename(path)), img.astype(np.float32))
