from filters import label
import numpy as np
from track_utils import prepare_costmat
from _munkres import munkres
from collections import Counter


def labels_map(lb0, lb1):
    """lb0 and lb1 should have objects in the same locations but different labels.
    """
    lbnums = np.unique(lb0).tolist()
    lbnums.remove(0)
    st = []
    for n0 in lbnums:
        n_sets = set(lb1[lb0 == n0])
        assert not len(n_sets) == 0
        n1 = list(n_sets)[0]
        st.append((n0, n1))
    return st


def convert_labels(lb_ref_to, lb_ref_from, lb_convert):
    """ Maps a reference set of labels onto another set of labels for the same image 
         Example: Separate segmentation of two objects in an image (nucleus and cytoplasm) leads
         to different labels and can reconcile the same cell to have the same label for both 
         objects. 
    Args:
        lb_ref_to (np.ndarray): image with object labels to convert to 
        lb_ref_from (np.ndarray): image with object labels to be converted 
        lb_convert (np.ndarray): the labeled image to be converted

    Returns:
        arr (numpy.ndarray): converted labels for objects with reference set of labels 

    """
    lbmap_to, lbmap_from = zip(*labels_map(lb_ref_to, lb_ref_from))
    arr = np.zeros(lb_convert.shape, dtype=np.uint16)
    lb = lb_convert.copy()
    for n0, n1 in zip(lbmap_to, lbmap_from):
        arr[lb == n0] = n1
    return arr


def convert_labels_lap(lb0, lb1, THRES=10):
    """convert values in lb1 to which corresponds to lb0.
    It calculates overlapping regions and solve a linear assignment problems.
    If overlap pixel is less than THRES, the object may be ignored.
    """
    
    id0 = list(np.unique(lb0))
    id0.remove(0)
    id1 = list(np.unique(lb1))
    id1.remove(0)
    co = []
    for i in id0:
        ovlap = lb1[lb0 == i]
        ovlap = ovlap[ovlap != 0]
        co.append(Counter(ovlap))

    cost = np.ones((len(id0), len(id1))) * np.Inf
    for n, c in enumerate(co):
        for k, v in c.iteritems():
            cost[n, id1.index(k)] = -v
    costmat = prepare_costmat(cost, -THRES, -THRES)
    t = munkres(costmat)
    topleft = t[0:cost.shape[0], 0:cost.shape[1]]

    idx = np.where(topleft)
    convto = [id0[a] for a in idx[0]]
    convfrom = [id1[a] for a in idx[1]]

    arr = np.zeros(lb0.shape)
    for ct, cf in zip(convto, convfrom):
        arr[lb1 == cf] = ct
    return arr

