from utils.subdetect_utils import dilate_to_cytoring, dilate_to_cytoring_buffer
from utils.concave_seg import levelset_geo_separete
from utils.concave_seg import run_concave_cut
from utils.filters import MultiSnakesCombined
from utils.concave_seg import levelset_lap
from utils.filters import label, adaptive_thresh
from scipy.ndimage.filters import minimum_filter


def ring_dilation(labels, MARGIN=0, RINGWIDTH=4):
    """Create a ring around label.
    :param RINGWIDTH (int): Width of rings
    :param MARGIN (int): A region of rings is ignored if they are within MARGIN pixels away from label.

    Examples:
        >>> arr = np.zeros((5, 5));arr[2, 2] = 10
        >>> ring_dilation(None, arr, None, MARGIN=1, RINGWIDTH=2)
        array([[ 0, 10, 10, 10,  0],
               [10,  0,  0,  0, 10],
               [10,  0,  0,  0, 10],
               [10,  0,  0,  0, 10],
               [ 0, 10, 10, 10,  0]], dtype=uint16)
    """
    return dilate_to_cytoring(labels, RINGWIDTH, MARGIN)


def ring_dilation_buffer(labels, MARGIN=0, RINGWIDTH=4, BUFFER=2):
    return dilate_to_cytoring_buffer(labels, RINGWIDTH, MARGIN, BUFFER)


def ring_dilation_above_thres(labels, img,  MARGIN=2, RINGWIDTH=4,
                              EXTRA_RINGWIDTH=15, THRES=50):
    sub_label = dilate_to_cytoring(labels, RINGWIDTH, MARGIN)
    extra_sub_label = dilate_to_cytoring(labels, EXTRA_RINGWIDTH, RINGWIDTH)
    extra_sub_label[img < THRES] = 0
    return sub_label + extra_sub_label


def ring_dilation_above_offset_buffer(labels, img, MARGIN=0, RINGWIDTH=2, BUFFER=2,
                                      OFFSET=100, FILSIZE=50):
    """Dilate from label to make a ring.
    Calculate the local minimum as a background, and if image is less brighter
    than background + offset, remove the region from the ring.
    """
    sub_label = dilate_to_cytoring_buffer(labels, RINGWIDTH, MARGIN, BUFFER)
    minimg = minimum_filter(img, size=FILSIZE)
    sub_label[img < (minimg + OFFSET)] = 0
    return sub_label


def ring_dilation_above_adaptive(labels, img, MARGIN=0, RINGWIDTH=4, BUFFER=2, RATIO=1.05, FILSIZE=10):
    sub_labels = dilate_to_cytoring_buffer(labels, RINGWIDTH, MARGIN, BUFFER)
    bw = adaptive_thresh(img, R=RATIO, FILTERINGSIZE=FILSIZE)
    sub_labels[-bw] = 0
    return sub_labels


def geodesic_levelset(labels, img, NITER=10, PROP=1):
    """propagated outwards until it sticks to the shape boundaries in an image.
    Generally used for making/repairing too small objects bigger.
    Larger NITER will give more propagation and roundy object.
    """
    return levelset_geo_separete(img, labels, niter=NITER, prop=PROP)


def concave_cut(labels, img, SMALL_RAD=7, LARGE_RAD=14, EDGELEN=6, THRES=180):
    """
    Attempt a cut for objects larger than np.pi * large_rad ** 2.
    For each pixel, the angle of two vectors pointing to boundary pixels distant by EDGELEN
    is calculated to find strong concavity more than THRES angle.
    The cut line is chosen by minimizing intensity below a line between two pixels.
    To reduce calculation costs, only pixels on a watershed line are assessed.

        SMALL_RAD (int): minimum radius of nuclei
        LARGE_RAD (int): maximum radius of nuclei
        EDGELEN (int): length of triangle edges to calculate angle
        THRES (int): threshold for concavity angles
    """
    labels = run_concave_cut(img, labels, small_rad=SMALL_RAD, large_rad=LARGE_RAD,
                             EDGELEN=EDGELEN, THRES=THRES)
    return labels


# def watershed_cut(labels, img, MIN_SIGMA=2, MAX_SIGMA=10, THRES=1000):
#     from utils.filters import lap_local_max, sitk_watershed_intensity
#     sigma_list = range(int(MIN_SIGMA), int(MAX_SIGMA))
#     local_maxima = lap_local_max(img, sigma_list, THRES)
#     return sitk_watershed_intensity(labels, local_maxima)


def propagate_multisnakes(labels, img, NITER=3, SMOOTHING=1, lambda1=1, lambda2=1):
    """
    Higher lambda2 relative to lambda1 gives more outward propagation.
    """
    ms = MultiSnakesCombined(img, labels, smoothing=SMOOTHING, lambda1=lambda1, lambda2=lambda2)
    labels = ms.multi_step(niter=NITER)
    return labels


def laplacian_levelset(labels, img, NITER=100, CURVE=3, PROP=-1):
    return label(levelset_lap(img, labels, NITER, CURVE, PROP))
