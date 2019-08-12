from __future__ import division
from skimage.measure import label
from utils.filters import adaptive_thresh
from skimage.filters import threshold_otsu
from utils.filters import label_watershed
from utils.binary_ops import grey_dilation
import numpy as np
from scipy.ndimage import gaussian_laplace, binary_dilation, binary_opening, binary_closing
from skimage.morphology import remove_small_objects
from utils.labels_handling import seeding_separate
from skimage.morphology import watershed

np.random.seed(0)


def constant_thres(img, THRES=2000, NEG=False):
    """take pixel above THRES as a foreground.

    Examples:
        >>> img = np.zeros((3, 3))
        >>> img[0, 0] = 10
        >>> img[2, 2] = 10
        >>> example_thres(img, None, THRES=5)
        array([[1, 0, 0],
               [0, 0, 0],
               [0, 0, 2]])
    """
    if NEG:
        return label(img < THRES)
    return label(img > THRES)


def global_otsu(img):
    global_thresh = threshold_otsu(img)
    return label(img > global_thresh)


def adaptive_thres(img, FIL1=10, R1=100):
    """adaptive thresholding for picking objects with different brightness.
    """
    bw = adaptive_thresh(img, R=R1, FILTERINGSIZE=FIL1)
    return label(bw)


def adaptive_thres_two(img, FIL1=10, FIL2=100, R1=100, R2=100):
    """adaptive thresholding for picking objects with different brightness.
    FIL2 and R2 for removing background.
    """
    bw = adaptive_thresh(img, R=R1, FILTERINGSIZE=FIL1)
    foreground = adaptive_thresh(img, R=R2, FILTERINGSIZE=FIL2)
    bw[-foreground] = 0
    return label(bw)


def adaptive_thres_otsu(img, FIL1=4, R1=1):
    """adaptive thresholding for picking objects with different brightness.
    Use Otsu's method for removing background
    """
    bw = adaptive_thresh(img, R1, FIL1)
    foreground = global_otsu(img) > 0
    bw[-foreground] = 0
    return label(bw)


def watershed_labels(labels, REG=10):
    """watershed to separate objects with concavity.
    Does not use intensity information but shape.
    """
    nlabels = label_watershed(labels.copy(), regmax=REG)
    nlabels[labels == 0] = 0
    return nlabels


def lap_peak_local(img, separation=10, percentile=64, min_sigma=2, max_sigma=5, num_sigma=10):
    sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
    gl_images = [-gaussian_laplace(img, s) * s ** 2 for s in sigma_list]
    image_cube = np.dstack(gl_images)
    max_image = np.max(image_cube, axis=2)
    coords = grey_dilation(max_image, separation=separation, percentile=percentile)

    def mark_pos(im, coords):
        temp = np.zeros(im.shape)
        for c0, c1 in coords:
            temp[c0, c1] = 1
        return temp
    bw = mark_pos(img, coords)
    return label(binary_dilation(bw, np.ones((3, 3))))


def agglomeration(img, MINSIZE=50, BG=50, INC=2.5):
    """
    INC: smaller it is, more resolution and computation
    BG: Threshold for background
    """
    li = []
    img = img.astype(np.float32)

    rs = np.arange(100, 0, -INC)
    perclist = []
    for r in rs:
        sc = np.percentile(img, r)
        if sc > BG:
            perclist.append(sc)
        else:
            break
    
    for _r in perclist:
        thresed = remove_small_objects(img > _r, MINSIZE, connectivity=2) > 0
        li.append(thresed.astype(np.uint16))
    q = np.sum(np.dstack(li), axis=2)
    p = label(q)
    for n, ind in enumerate(range(int(np.max(q)-1), 0, -1)):
        c = seeding_separate(label(q >= ind), p)
        w = watershed(q >= ind, markers=c, mask=(q>=ind), watershed_line=True)
        p = label(w, connectivity=2)
    return p