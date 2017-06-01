"""

Label labels1 to the same value from labels0. Not tracked is negative.
Turn labels0 into negative.

"""


from utils.filters import label
from utils.postprocess_utils import regionprops
from scipy.spatial.distance import cdist
from utils.track_utils import calc_massdiff, find_one_to_one_assign
from utils.global_holder import holder  # holder is used to store parameters that needs to be called many times
import numpy as np
from utils.track_utils import call_lap, pick_closer_cost
from utils.filters import labels2outlines
from utils.concave_seg import wshed_raw, CellCutter
from utils.track_utils import _find_best_neck_cut, _update_labels_neck_cut
from utils.global_holder import holder
from scipy.ndimage import gaussian_laplace, binary_dilation
from utils.binary_ops import grey_dilation
np.random.seed(0)


def nearest_neighbor(img0, img1, labels0, labels1, DISPLACEMENT=20, MASSTHRES=0.2):
    """
    labels0 and labels1: the positive values for non-tracked objects and the negative values for tracked objects.
    """
    labels = -labels1.copy()
    rps0 = regionprops(labels0, img0)
    rps1 = regionprops(labels1, img1)
    if not rps0 or not rps1:
        return labels0, labels
    dist = cdist([i.centroid for i in rps0], [i.centroid for i in rps1])
    massdiff = calc_massdiff(rps0, rps1)
    binary_cost = (dist < DISPLACEMENT) * (abs(massdiff) < MASSTHRES)
    idx1, idx0 = find_one_to_one_assign(binary_cost)
    for i0, i1 in zip(idx0, idx1):
        labels[labels1 == rps1[i1].label] = rps0[i0].label
        labels0[labels0 == rps0[i0].label] = -rps0[i0].label
    return labels0, labels


def nn_closer(img0, img1, labels0, labels1, DISPLACEMENT=30, MASSTHRES=0.25):
    labels = -labels1.copy()
    rps0 = regionprops(labels0, img0)
    rps1 = regionprops(labels1, img1)

    if not rps0 or not rps1:
        return labels0, labels
    dist = cdist([i.centroid for i in rps0], [i.centroid for i in rps1])
    massdiff = calc_massdiff(rps0, rps1)
    binary_cost = (dist < DISPLACEMENT) * (abs(massdiff) < MASSTHRES)
    binary_cost = pick_closer_cost(binary_cost, dist)
    binary_cost = pick_closer_cost(binary_cost.T, dist.T).T
    idx1, idx0 = find_one_to_one_assign(binary_cost)
    for i0, i1 in zip(idx0, idx1):
        labels[labels1 == rps1[i1].label] = rps0[i0].label
        labels0[labels0 == rps0[i0].label] = -rps0[i0].label
    return labels0, labels


def run_lap(img0, img1, labels0, labels1, DISPLACEMENT=30, MASSTHRES=0.2):
    '''Linear assignment problem for mammalian cells.
    Cost matrix is simply the distance.
    costDie and costBorn are variables changing over frame. Update it through holder.

    Args:
    DISPLACEMENT (int): The maximum distance (in pixel)
    MASSTHRES (float):  The maximum difference of total intensity changes.
                        0.2 means it allows for 20% total intensity changes.
    '''
    labels = -labels1.copy()
    rps0 = regionprops(labels0, img0)
    rps1 = regionprops(labels1, img1)

    if not rps0 or not rps1:
        return labels0, labels

    dist = cdist([i.centroid for i in rps0], [i.centroid for i in rps1])
    massdiff = calc_massdiff(rps0, rps1)

    '''search radius is now simply set by maximum displacement possible.
    In the future, I might add data-driven approach mentioned in LAP paper (supple pg.7)'''
    dist[dist > DISPLACEMENT] = np.Inf  # assign a large cost for unlikely a pair
    # dist[abs(massdiff) > MASSTHRES] = np.Inf
    cost = dist
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return labels1

    # Define initial costBorn and costDie in the first frame
    if not hasattr(holder, 'cost_born') or not hasattr(holder, 'cost_die'):
        holder.cost_born = np.percentile(cost[~np.isinf(cost)], 80)
        holder.cost_die = np.percentile(cost[~np.isinf(cost)], 80)
    # try:
    binary_cost = call_lap(cost, holder.cost_die, holder.cost_born)
    # The first assignment of np.Inf is to reduce calculation of linear assignment.
    # This part will make sure that cells outside of these range do not get connected.
    binary_cost[(np.abs(massdiff) > MASSTHRES)] = False
    binary_cost[(dist > DISPLACEMENT)] = False

    gp, gc = np.where(binary_cost)
    idx0, idx1 = list(gp), list(gc)

    for i0, i1 in zip(idx0, idx1):
        labels[labels1 == rps1[i1].label] = rps0[i0].label
        labels0[labels0 == rps0[i0].label] = -rps0[i0].label

    # update cost
    linked_dist = [dist[i0, i1] for i0, i1 in zip(idx0, idx1)]
    if linked_dist:
        cost = np.max(linked_dist)*1.05
        if cost != 0:  # solver freezes if cost is 0
            holder.cost_born, holder.cost_die = cost, cost
    return labels0, labels


def track_neck_cut(img0, img1, labels0, labels1, DISPLACEMENT=10, MASSTHRES=0.2,
                   EDGELEN=5, THRES_ANGLE=180, WSLIMIT=False):
    """
    Adaptive segmentation by using tracking informaiton.
    Separate two objects by making a cut at the deflection. For each points on the outline,
    it will make a triangle separated by EDGELEN and calculates the angle facing inside of concave.

    The majority of cells need to be tracked before the this method to calculate LARGE_RAD and SMALL_RAD.

    EDGELEN (int):      A length of edges of triangle on the nuclear perimeter.
    THRES_ANGLE (int):  Define the neck points if a triangle has more than this angle.
    STEPLIM (int):      points of neck needs to be separated by at least STEPLIM in parimeters.
    WSLIMIT (bool):     Limit search points to ones overlapped with watershed transformed images. Set it True if calculation is slow.
    """
    # labels0, labels = nn_closer(img0, img1, labels0, labels1, DISPLACEMENT, MASSTHRES)
    # labels1 = -labels.copy()
    CANDS_LIMIT = 300
    labels = -labels1.copy()

    if not hasattr(holder, "SMALL_RAD") and not hasattr(holder, "LARGE_RAD"):
        tracked_area = [i.area for i in regionprops(labels)]
        LARGE_RAD = np.sqrt(max(tracked_area)/np.pi)
        SMALL_RAD = np.sqrt(np.percentile(tracked_area, 5)/np.pi)
        holder.LARGE_RAD = LARGE_RAD
        holder.SMALL_RAD = SMALL_RAD
    else:
        SMALL_RAD = holder.SMALL_RAD
        LARGE_RAD = holder.LARGE_RAD

    rps0 = regionprops(labels0, img0)
    unique_labels = np.unique(labels1)

    if WSLIMIT:
        wlines = wshed_raw(labels1 > 0, img1)
    else:
        wlines = np.ones(labels1.shape, np.bool)

    store = []
    coords_store = []
    for label_id in unique_labels:
        if label_id == 0:
            continue
        cc = CellCutter(labels1 == label_id, img1, wlines, small_rad=SMALL_RAD,
                        large_rad=LARGE_RAD, EDGELEN=EDGELEN, THRES=THRES_ANGLE)
        cc.prepare_coords_set()
        candidates = cc.search_cut_candidates(cc.bw.copy(), cc.coords_set[:CANDS_LIMIT])
        for c in candidates:
            c.raw_label = label_id
        store.append(candidates)
        coords_store.append(cc.coords_set)
    coords_store = [i for i in coords_store if i]
    # Attempt a first cut.
    good_cells = _find_best_neck_cut(rps0, store, DISPLACEMENT, MASSTHRES)
    labels0, labels = _update_labels_neck_cut(labels0, labels1, good_cells)

    # iteration from here.
    while good_cells:
        rps0 = regionprops(labels0, img0)
        labels1 = -labels.copy()
        rps0 = regionprops(labels0, img0)
        unique_labels = np.unique(labels1)

        store = []
        for label_id in unique_labels:
            if label_id == 0:
                continue
            bw = labels1 == label_id
            coords_set = [i for i in coords_store if bw[i[0][0][0], i[0][0][1]]]
            if not coords_set:
                continue
            coords_set = coords_set[0]
            candidates = cc.search_cut_candidates(bw, coords_set)
            for c in candidates:
                c.raw_label = label_id
            store.append(candidates)
            coords_store.append(coords_set)
        good_cells = _find_best_neck_cut(rps0, store, DISPLACEMENT, MASSTHRES)
        labels0, labels = _update_labels_neck_cut(labels0, labels1, good_cells)
    labels0, labels = nn_closer(img0, img1, labels0, -labels, DISPLACEMENT, MASSTHRES)
    return labels0, labels

