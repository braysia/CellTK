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


def nearest_neighbor(img0, img1, labels0, labels1, DISPLACEMENT=20, MASSTHRES=0.2):
    labels = -labels1.copy()
    rps0 = regionprops(labels0, img0)
    rps1 = regionprops(labels1, img1)

    dist = cdist([i.centroid for i in rps0], [i.centroid for i in rps1])
    massdiff = calc_massdiff(rps0, rps1)
    binary_cost = (dist < DISPLACEMENT) * (abs(massdiff) < MASSTHRES)
    idx1, idx0 = find_one_to_one_assign(binary_cost)
    for i0, i1 in zip(idx0, idx1):
        labels[labels1 == rps1[i1].label] = rps0[i0].label
        labels0[labels0 == rps0[i0].label] = -rps0[i0].label
    return labels0, labels


def run_lap(img0, img1, labels0, labels1, DISPLACEMENT=20, MASSTHRES=0.2):
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
    binary_cost[abs(massdiff) > MASSTHRES] = False
    binary_cost[dist > DISPLACEMENT] = False

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


def track_neck_cut(img0, img1, labels0, labels1, ERODI=5, DEBRISAREA=50, DISPLACEMENT=30,
                   MASSTHRES=0.2, LIM=10, EDGELEN=5, THRES_ANGLE=180, STEPLIM=10):
        """
        Adaptive segmentation by using tracking informaiton.
        Separate two objects by making a cut at the deflection. For each points on the outline,
        it will make a triangle separated by EDGELEN and calculates the angle facing inside of concave.

        EDGELEN (int):      A length of edges of triangle on the nuclear perimeter.
        THRES_ANGLE (int):  Define the neck points if a triangle has more than this angle.
        STEPLIM (int):      points of neck needs to be separated by at least STEPLIM in parimeters.
        """
        labels = -labels1.copy()

        rps0 = regionprops(labels0, img0)
        unique_labels = np.unique(labels1)

        wlines = wshed_raw(labels1 > 0, img1)
        store = []
        coords_store = []
        for label_id in unique_labels:
            if label_id == 0:
                continue
            cc = CellCutter(labels1 == label_id, img1, wlines, small_rad=7, large_rad=20)
            cc.prepare_coords_set()
            candidates = cc.search_cut_candidates(cc.bw.copy(), cc.coords_set)
            for c in candidates:
                c.raw_label = label_id
            store.append(candidates)
            coords_store.append(cc.coords_set)

        # Find good cells from candidates.
        # Update labels. Update coords_set.
        # Repeat.
        # two candidates have the same line intensity.
        good_cells = []
        for cands in store:
            if not cands or not rps0:
                continue
            # cands = sorted(cands, key=lambda x: x.line_total)
            dist = cdist([i.centroid for i in rps0], [i.centroid for i in cands])
            massdiff = calc_massdiff(rps0, cands)
            binary_cost = (dist < DISPLACEMENT) * (abs(massdiff) < MASSTHRES)

            line_int = [i.line_total for i in cands]
            line_mat = np.tile(line_int, len(rps0)).reshape(len(rps0), len(line_int))
            binary_cost = pick_closer_cost(binary_cost, line_mat)  # pick anything closer in mass
            binary_cost = pick_closer_cost(binary_cost.T, dist.T).T  # pick anything closer in mass
            idx1, idx0 = find_one_to_one_assign(binary_cost.copy())
            if not idx0:
                continue
            i0, i1 = idx0[0], idx1[0]
            cell = cands[i1]
            cell.previous = rps0[i0]
            good_cells.append(cell)

        minint = -np.max(abs(labels))
        unique_raw_labels = np.unique([cell.raw_label for cell in good_cells])
        neg_labels = np.zeros(labels.shape, np.int32)
        for i in unique_raw_labels:
            minint -= 1
            neg_labels[labels1 == i] = minint
            labels[labels1 == i] = 0

        for cell in good_cells:
            for c0, c1 in cell.coords:
                neg_labels[c0, c1] = cell.previous.label
        labels = labels + neg_labels
        return labels0, labels
