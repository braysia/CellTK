from utils.subdetect_utils import dilate_to_cytoring


def ring_dilation(labels, MARGIN=0, RINGWIDTH=4):
    """Create a ring around label.
    :param RINGWIDTH: Width of rings
    :param MARGIN: A region of rings is ignored if they are within MARGIN pixels away from label.

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