import os
import imp
import numpy as np
from scipy.ndimage import imread as imread0
import tifffile as tiff



def conv_labels2dto3d(labels):
    lbnums = np.unique(labels)
    arr = np.zeros((labels.shape[0], labels.shape[1], len(lbnums)), np.uint8)
    for i in lbnums:
        arr[:, :, i] = labels == i
    return arr


def normalize(orig_img):
    percentile = 99.9
    high = np.percentile(orig_img, percentile)
    low = np.percentile(orig_img, 100-percentile)
    img = np.minimum(high, orig_img)
    img = np.maximum(low, img)
    img = (img - low) / (high - low)
    return img


def make_outputdir(output):
    try:
        os.makedirs(output)
    except:
        pass


def imread_check_tiff(path):
    img = imread0(path)
    if img.dtype == 'object' or path.endswith('tif'):
        img = tiff.imread(path)
    return img


def imread(path):
    if isinstance(path, tuple) or isinstance(path, list):
        st = []
        for p in path:
            st.append(imread_check_tiff(p))
        img = np.dstack(st)
        if img.shape[2] == 1:
            np.squeeze(img, axis=2)
        return img
    else:
        return imread_check_tiff(path)


def parse_image_files(inputs):
    if "/" not in inputs:
        return (inputs, )
    store = []
    li = []
    while inputs:
        element = inputs.pop(0)
        if element == "/":
            store.append(li)
            li = []
        else:
            li.append(element)
    store.append(li)
    return zip(*store)


def pad_image(image):
    # assumes the image is an np array of dimensions d0 x d1 x d2 x d3
    # where d1 is height, d2 is width, and d3 is colors
    # returns a list of padded image, hpadding (tuple), wpadding (tuple)

    height = image.shape[1]
    hdelta = 8 - (height % 8)
    if (hdelta == 8):
        hpadding = (0, 0)
    elif (hdelta % 2) == 0:
        hpadding = (int(hdelta/2.0), int(hdelta/2.0))
    else:
        hpadding = (int(hdelta/2.0), int(hdelta/2.0)+1)

    width = image.shape[2]
    wdelta = 8 - (width % 8)
    if (wdelta == 8):
        wpadding = (0, 0)
    elif (wdelta % 2) == 0:
        wpadding = (int(wdelta/2.0), int(wdelta/2.0))
    else:
        wpadding = (int(wdelta/2.0), int(wdelta/2.0)+1)
    return [np.pad(image, ((0, 0), hpadding, wpadding, (0, 0)), 'constant', constant_values=0.0), hpadding, wpadding]


def normalize_predictions(predictions):
    # predictions is typically a 3 x h x w list
    predictions = np.array(predictions)
    num_rows = predictions.shape[1]
    num_cols = predictions.shape[2]

    for i in range(num_rows):
        for j in range(num_cols):
            prediction = predictions[:, i, j]
            if (prediction.sum() != 0):
                predictions[:, i, j] = prediction / prediction.sum()

    return predictions



