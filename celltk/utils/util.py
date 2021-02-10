import numpy as np
from scipy.ndimage import imread as imread0
import tifffile as tiff
import skimage
from skimage import io

def imread_check_tiff(path,four_d=False):
    if four_d:
        img = io.imread(path)
        img = img[:,:,0,:]
    else:
        img = imread0(path)
    if img.dtype == 'object':
        img = tiff.imread(path)
    return img


def imread(path,four_d=False):
    if isinstance(path, tuple) or isinstance(path, list):
        st = []
        for p in path:
            st.append(imread_check_tiff(p))
        img = np.dstack(st)
        if img.shape[2] == 1:
            np.squeeze(img, axis=2)
        return img
    else:
        if 'op003' in path:
            if 'processed' not in path:
                four_d = True 
        return imread_check_tiff(path,four_d)