import numpy as np
from scipy.optimize import linear_sum_assignment


def munkres(arr):
    temp = np.zeros(arr.shape, np.bool)
    ind = linear_sum_assignment(arr)
    temp[ind] = True
    return temp