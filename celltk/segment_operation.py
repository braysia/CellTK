from skimage.measure import label
from utils.filters import adaptive_thresh
from skimage.filters import threshold_otsu
from utils.filters import label_watershed


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


def adaptive_thres(img, FIL1=10, FIL2=100, R1=1, R2=1):
    """adaptive thresholding for picking objects with different brightness.
    FIL2 and R2 for removing background.
    """
    bw = adaptive_thresh(img, FIL1)
    foreground = adaptive_thresh(img, FIL2)
    bw[-foreground] = 0
    return label(bw)


def adaptive_thres_two(img, FIL1=4, FIL2=100, R1=1, R2=1):
    """need img.shape==(n, m, 2)
    """
    bw = adaptive_thresh(img[:, :, 0], R1, FIL1)
    foreground = adaptive_thresh(img[:, :, 1], FIL2)
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
    return label_watershed(labels, regmax=REG)

