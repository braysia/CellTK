from skimage.measure import label
from utils.postprocess_utils import regionprops
from scipy.spatial.distance import cdist
from utils.track_utils import calc_massdiff, find_one_to_one_assign
from global_holder import holder
import numpy as np
from utils.track_utils import call_lap


def nearest_neighbor(img0, img1, labels0, labels1, DISPLACEMENT=50, MASSTHRES=0.2):
    labels = -labels1.copy()
    rps0 = regionprops(labels0, img0)
    rps1 = regionprops(labels1, img1)

    dist = cdist([i.centroid for i in rps0], [i.centroid for i in rps1])
    massdiff = calc_massdiff(rps0, rps1)
    binary_cost = (dist < DISPLACEMENT) * (abs(massdiff) < MASSTHRES)
    idx1, idx0 = find_one_to_one_assign(binary_cost)
    for i0, i1 in zip(idx0, idx1):
        labels[labels1 == rps1[i1].label] = rps0[i0].label
    return labels


def run_lap(img0, img1, labels0, labels1, DISPLACEMENT=100, MASSTHRES=0.2):
    '''Linear assignment problem for mammalian cells.
    Cost matrix is simply the distance.
    costDie and costBorn are variables changing over frame. Update it through holder.

    Args:
    DISPLACEMENT (int): The maximum distance (in pixel)
    MASSTHRES (float):  The maximum difference of total intensity changes.
                        0.2 means it allows for 20% total intensity changes.
    '''
    labels = labels1.copy()
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

    # update cost
    linked_dist = [dist[i0, i1] for i0, i1 in zip(idx0, idx1)]
    if linked_dist:
        cost = np.max(linked_dist)*1.05
        if cost != 0:  # solver freezes if cost is 0
            holder.cost_born, holder.cost_die = cost, cost
    return labels
