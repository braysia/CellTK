from __future__ import division
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.measure import label as skim_label
from skimage.morphology import watershed as skiwatershed
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries
from skimage.feature import peak_local_max
from scipy.ndimage.filters import maximum_filter
from skimage.draw import line
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk


def watershed(labels, regmax):
    # Since there are non-unique values for dist, add very small numbers. This will separate each marker by regmax at least.
    dist = distance_transform_edt(labels) + np.random.rand(*labels.shape)*1e-10
    labeled_maxima = label(peak_local_max(dist, min_distance=int(regmax), indices=False))
    wshed = -dist
    wshed = wshed - np.min(dist)
    markers = np.zeros(wshed.shape, np.int16)
    markers[labeled_maxima>0]= -labeled_maxima[labeled_maxima>0]
    wlabel = skiwatershed(wshed, markers, connectivity=np.ones((3,3), bool), mask=labels!=0)
    wlabel = -wlabel
    wlabel = labels.max() + wlabel
    wlabel[wlabel == labels.max()] = 0
    all_label = label(labels + wlabel)
    return all_label


def label(bw, connectivity=2):
    '''original label might label any objects at top left as 1. To get around this pad it first.'''
    if bw[0, 0]:
        return skim_label(bw, connectivity=connectivity)
    bw = np.pad(bw, pad_width=1, mode='constant', constant_values=False)
    labels = skim_label(bw, connectivity=connectivity)
    labels = labels[1:-1, 1:-1]
    return labels


def peak_local_max_edge(labels, min_dist=5):
    '''peak_local_max sometimes shows a weird behavior...?'''
    label_max = maximum_filter(labels, size=min_dist)
    mask = label == label_max
    label[-mask] = 0
    return labels


def find_label_boundaries(labels):
    blabels = labels.copy()
    bwbound = find_boundaries(blabels)
    blabels[-bwbound] = 0
    return blabels


def labels2outlines(labels):
    """Same functionality with find_label_boundaries.
    """
    outlines = labels.copy()
    outlines[-find_boundaries(labels)] = 0
    return outlines


def adaptive_thresh(img, RATIO=3.0, FILTERINGSIZE=50):
    """Segment as a foreground if pixel is higher than ratio * blurred image.
    If you set ratio 3.0, it will pick the pixels 300 percent brighter than the blurred image.
    """
    fim = gaussian_filter(img, FILTERINGSIZE)
    bw = img > (fim * RATIO)
    return bw


def calc_lapgauss(img, SIGMA=2.5):
    fil = sitk.LaplacianRecursiveGaussianImageFilter()
    fil.SetSigma(SIGMA)
    # fil.SetNormalizeAcrossScale(False)
    csimg = sitk.GetImageFromArray(img)
    slap = fil.Execute(csimg)
    return sitk.GetArrayFromImage(slap)
