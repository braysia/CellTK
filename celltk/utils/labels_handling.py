from filters import label
import numpy as np


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


def seeding_separate(c, p):
    """
    Keep the markers separated if there are two previous markers on one current marker.
    Assuming foreground in c is always including foreground in p. 
    """
    dou_c = []
    dou_p = []
    both = [i for i in list(set(p[c > 0])) if not i == 0]  # exists in previous and current. 
    for b in both:
        sp = [i for i in set(p[c==b]) if not i == 0]
        if len(sp) > 1:
            dou_p.append(sp)
            dou_c.append(b)

    for dc in dou_c:
        c[c==dc] = 0
    c += 10000  # dirty implementation
    c[c==10000] = 0
    dou_p = [i for j in dou_p for i in j]
    for dp in dou_p:
        c[p == dp] = dp
    return c

