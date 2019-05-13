from utils.subdetect_utils import dilate_to_cytoring, dilate_to_cytoring_buffer
from utils.concave_seg import levelset_geo_separete
from utils.concave_seg import run_concave_cut
from utils.filters import MultiSnakesCombined
from utils.concave_seg import levelset_lap
from utils.filters import label, adaptive_thresh
from scipy.ndimage.filters import minimum_filter
import numpy as np
from scipy.ndimage import morphology
from skimage.morphology import remove_small_objects
from utils.labels_handling import convert_labels
from utils.subdetect_utils import label_high_pass, label_nearest, repair_sal, label_nearest_or_within
from utils.global_holder import holder
from segment_operation import constant_thres
np.random.seed(0)


def ring_dilation(labels, MARGIN=0, RINGWIDTH=4, BUFFER=0):
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
    if BUFFER == 0:
        return dilate_to_cytoring(labels, RINGWIDTH, MARGIN)
    else:
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


def watershed_cut(labels, img, MIN_SIGMA=2, MAX_SIGMA=10, THRES=1000):
    from utils.filters import lap_local_max, sitk_watershed_intensity
    sigma_list = range(int(MIN_SIGMA), int(MAX_SIGMA))
    local_maxima = lap_local_max(img, sigma_list, THRES)
    return sitk_watershed_intensity(labels, local_maxima)


def propagate_multisnakes(labels, img, NITER=3, SMOOTHING=1, lambda1=1, lambda2=1, keep=False,no_shrink=False):
    """
    Higher lambda2 relative to lambda1 gives more outward propagation.
    Setting keep=True will try to keep the values in labels but may slow down computation.
    Setting no_shrink=True will ensure that the mask can only stay the same shape or grow larger 
    """
    labels_orig = labels.copy()
    ms = MultiSnakesCombined(img, labels, smoothing=SMOOTHING, lambda1=lambda1, lambda2=lambda2, keep=keep)
    labels = ms.multi_step(niter=NITER)
    if no_shrink:
        comb = np.max(np.dstack((labels, labels_orig)), axis=2).astype(np.uint16)
        return comb
    else:
        return labels


def laplacian_levelset(labels, img, NITER=100, CURVE=3, PROP=-1):
    return label(levelset_lap(img, labels, NITER, CURVE, PROP))


def voronoi_cut(labels):
    from utils.subdetect_utils import voronoi_expand
    return voronoi_expand(labels)


def detect_puncta_voronoi(labels, img, level=7, PERC=50, FILSIZE=1):
    from utils.subdetect_utils import voronoi_expand
    from utils.fish_detect import detect_puncta
    vor = voronoi_expand(labels)
    puncta = detect_puncta(img, level=level, PERC=PERC, FILSIZE=FILSIZE)
    vor[puncta == 0] = 0
    return vor


def morphological(labels, func='grey_opening', size=3, iterations=1):
    morph_operation = getattr(morphology, func)
    for i in range(iterations):
        labels = morph_operation(labels, size=(size, size))
    return labels


def watershed_divide(labels, regmax=10, min_size=100):
    """
    divide objects in labels with watershed segmentation.
        regmax:
        min_size: objects smaller than this size will not be divided.
    """
    from utils.subdetect_utils import watershed_labels

    large_labels = remove_small_objects(labels, min_size, connectivity=4)
    labels[large_labels > 0] = 0
    ws_large = watershed_labels(large_labels, regmax)
    ws_large += labels.max()
    ws_large[ws_large == labels.max()] = 0
    return labels + ws_large


def cytoplasm_levelset(labels, img, niter=20, dt=-0.5, thres=0.5):
    """ Segment cytoplasm from supplied nuclear labels and probability map 
        Expand using level sets method from nuclei to membrane.
        Uses an implementation of Level set method. See: 
        https://wiseodd.github.io/techblog/2016/11/05/levelset-method/
    
    Args:
        labels (numpy.ndarray): nuclear mask labels
        img (numpy.ndarray): probability map 
        niter (int): step size to expand mask, number of iterations to run the levelset algorithm
        dt (float): negative values for porgation, positive values for shrinking 
        thres (float): threshold of probability value to extend the nuclear mask to
    
    Returns:
        cytolabels (numpy.ndarray): cytoplasm mask labels  

    """
    from skimage.morphology import closing, disk, remove_small_holes
    from utils.dlevel_set import dlevel_set
    phi = labels.copy()
    phi[labels == 0] = 1
    phi[labels > 0] = -1

    outlines = img.copy()
    outlines = -outlines
    outlines = outlines - outlines.min()
    outlines = outlines/outlines.max()

    mask = outlines < thres
    phi = dlevel_set(phi, outlines, niter=niter, dt=dt, mask=mask)

    cytolabels = label(remove_small_holes(label(phi < 0)))
    cytolabels = closing(cytolabels, disk(3))

    temp = cytolabels.copy()
    temp[labels == 0] = 0
    cytolabels = convert_labels(temp, labels, cytolabels)
    return cytolabels


def segment_bacteria(nuc, img, slen=3, SIGMA=0.5, THRES=20, CLOSE=20, MINAREA=5,
                     dist=25, SEGMENT='highpass', ASSIGN=True):
    """ Segment bacteria using high pass filter or constant thresholding and assign to closest nucleus

    Args:
        nuc (numpy.ndarray): nuclear mask labels
        img (numpy.ndarray): image in bacterial channel
        slen (int): Size of Gaussian kernel
        SIGMA (float): Standard deviation for Gaussian kernel
        THRES (int): Threshold pixel intensity fo real signal 
        CLOSE (int): Radius for disk used to return morphological closing of the image
                     (dilation followed by erosion to remove dark spots and connect bright cracks)
        THRESCHANGE (int): argument unnecessary? 
        MINAREA (int): minimum area in pixels for a bacterium 
        dist (int): acceptable distance bac can be from mask 
        SEGMENT: choose one of 'highpass' or 'constant'.
        ASSIGN: If False, it does not assign to closest nucleus. 
    Returns:
        labels (numpy.ndarray[np.uint16]): bacterial mask labels  

    """
    if SEGMENT == 'highpass':
        labels = label_high_pass(img, slen=slen, SIGMA=SIGMA, THRES=THRES, CLOSE=3)
    elif SEGMENT == 'constant':
        labels = constant_thres(img, THRES=THRES)
    if not ASSIGN:
        return labels.astype(np.uint16)
    if labels.any():
        labels, comb, nuc_prop, nuc_loc = label_nearest(img, labels, nuc, dist)
    from skimage.morphology import remove_small_objects
    labels = remove_small_objects(labels, MINAREA)
    return labels.astype(np.uint16)

def segment_bacteria_within_cyto(nuc, img, slen=3, SIGMA=0.5, THRES=20, CLOSE=20, MINAREA=5,
                     dist=25, SEGMENT='highpass', ASSIGN=True):
    """ Segment bacteria using high pass filter or constant thresholding and assign to closest nucleus

    Args:
        nuc (numpy.ndarray): nuclear mask labels
        img (numpy.ndarray): image in bacterial channel
        slen (int): Size of Gaussian kernel
        SIGMA (float): Standard deviation for Gaussian kernel
        THRES (int): Threshold pixel intensity fo real signal 
        CLOSE (int): Radius for disk used to return morphological closing of the image
                     (dilation followed by erosion to remove dark spots and connect bright cracks)
        THRESCHANGE (int): argument unnecessary? 
        MINAREA (int): minimum area in pixels for a bacterium 
        dist (int): acceptable distance bac can be from mask 
        SEGMENT: choose one of 'highpass' or 'constant'.
        ASSIGN: If False, it does not assign to closest nucleus. 
    Returns:
        labels (numpy.ndarray[np.uint16]): bacterial mask labels  

    """
    if SEGMENT == 'highpass':
        labels = label_high_pass(img, slen=slen, SIGMA=SIGMA, THRES=THRES, CLOSE=3)
    elif SEGMENT == 'constant':
        labels = constant_thres(img, THRES=THRES)
    if not ASSIGN:
        return labels.astype(np.uint16)
    if labels.any():
        labels, comb, nuc_prop, nuc_loc = label_nearest_or_within(img, labels, nuc, dist)
    from skimage.morphology import remove_small_objects
    labels = remove_small_objects(labels, MINAREA)
    return labels.astype(np.uint16)

def keep_certain_labels(nuc,img):
    """ Modify labels so that labels with a certain property (i.e. without bacteria) are removed.
        This function will remove nuclear labels that do not have an associated mask label 
        nuc (numpy.ndarray): nuclear mask labels
        img (numpy.ndarray): mask labels for another object 
    """
    labels_keep = np.unique(nuc)
    labels_keep = np.array(np.delete(labels_keep,0))
    ix = np.in1d(img.ravel(), labels_keep).reshape(img.shape)
    indices_remove = np.invert(ix)
    img[indices_remove] = 0 
    return img

def segment_bacteria_return_cyto_no_bac(nuc, img, slen=3, SIGMA=0.5,THRES=20, CLOSE=20, THRESCHANGE=1000, MINAREA=5, dist=25):
    """ Segment bacteria and assign to closest nucleus and return the mask back without regions containing bacteria

    Args:
        nuc (numpy.ndarray): nuclear mask labels
        img (numpy.ndarray): image in bacterial channel
        slen (int): Size of Gaussian kernel
        SIGMA (float): Standard deviation for Gaussian kernel
        THRES (int): Threshold pixel intensity fo real signal 
        CLOSE (int): Radius for disk used to return morphological closing of the image (dilation followed by erosion to remove dark spots and connect bright cracks)
        THRESCHANGE (int): argument unnecessary? 
        MINAREA (int): minimum area in pixels for a bacterium 

    Returns:
        labels (numpy.ndarray[np.uint16]): bacterial mask labels  

    """
    labels = label_high_pass(img, slen=slen, SIGMA=SIGMA, THRES=THRES, CLOSE=3)
    if labels.any():
        labels, comb, nuc_prop, nuc_loc = label_nearest(img, labels, nuc,dist)
    from skimage.morphology import remove_small_objects
    labels = remove_small_objects(labels, MINAREA)
    labels = comb - labels 
    return labels.astype(np.uint16)


def filter_with_labels(labels, img, BG=True):
    """
    Modify an image so that background regions in labels have value 0
    and the rest of the pixels have their original value.
    Setting BG to False will set foreground regions to be 0.
    """
    if BG:
        img[labels == 0] = 0
    if not BG:
        img[labels > 0] = 0
    return img



def segment_bacteria_repair(nuc, img, slen=3, SIGMA=0.5,THRES=20, CLOSE=20, THRESCHANGE=1000, MINAREA=5, dist=25):
    """ Segment bacteria and assign to closest nucleus ## This is still under construction

    Args:
        nuc (numpy.ndarray): nuclear mask labels
        img (numpy.ndarray): image in bacterial channel
        slen (int): Size of Gaussian kernel
        SIGMA (float): Standard deviation for Gaussian kernel
        THRES (int): Threshold pixel intensity fo real signal 
        CLOSE (int): Radius for disk used to return morphological closing of the image (dilation followed by erosion to remove dark spots and connect bright cracks)
        THRESCHANGE (int): argument unnecessary? 
        MINAREA (int): minimum area in pixels for a bacterium 

    Returns:
        labels (numpy.ndarray[np.uint16]): bacterial mask labels  

    """
    labels = label_high_pass(img, slen=slen, SIGMA=SIGMA, THRES=THRES, CLOSE=3)
    if labels.any():
        labels, comb, nuc_prop, nuc_loc = label_nearest(img, labels, nuc,dist)
        if not hasattr(holder,'plabel'): #not sure what this is doing
            holder.plabel = labels
            holder.pcomb = comb
            holder.pimg = img 
            return labels
            #.astype(np.unit16)
        labels = repair_sal(img, holder.pimg,comb, holder.pcomb,labels,nuc_prop,nuc_loc,THRESCHANGE)
    from skimage.morphology import remove_small_objects
    labels = remove_small_objects(labels, MINAREA)
    return labels
    #labels.astype(np.uint16)

def keep_best_prediction(cyto,img):
    print cyto.shape
    print img.shape
    if img.shape == cyto.shape:
        new_img = np.zeros((img.shape[0],img.shape[1]))
        x = img.shape[0]
        y = img.shape[1]
        for i in range(0,x):
            for j in range(0,y):
                print i
                img_val = img[i,j]
                label_val = cyto[i,j]
                if not img_val == label_val:
                    new_img[i,j] = img_val + label_val
                else:
                    new_img[i,j] = img_val 
        return new_img
    else:
        return



