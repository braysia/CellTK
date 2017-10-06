import numpy as np
from scipy.ndimage import imread as imread0
import tifffile as tiff


def imread_check_tiff(path):
    img = imread0(path)
    if img.dtype == 'object':
        img = tiff.imread(path)
    return img


def imread(path):
    if isinstance(path, tuple) or isinstance(path, list):
        st = []
        for p in path:
            st.append(imread_check_tiff(p))
        return np.dstack(st)
    else:
        return imread_check_tiff(path)