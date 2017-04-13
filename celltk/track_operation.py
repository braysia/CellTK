from skimage.measure import label
from utils.postprocess_utils import regionprops
from scipy.spatial.distance import cdist
from utils.track_utils import calc_massdiff, find_one_to_one_assign


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
