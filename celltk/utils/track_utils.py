from __future__ import division
import numpy as np


def calc_ratiodiff(a, b):
    """calculate how much pairwise ratio of vector a to b.
    """
    a0, a1 = np.meshgrid(a, b)
    return (a1.T - a0.T)/a0.T


def calc_diff(a, b):
    """calculate how much pairwise ratio of vector a to b.
    """
    a0, a1 = np.meshgrid(a, b)
    return a1.T - a0.T


def calc_massdiff(cells0, cells1):
    return calc_ratiodiff([i.total_intensity for i in cells0], [i.total_intensity for i in cells1])


def find_one_to_one_assign(cost):
    (_, col1) = np.where([np.sum(cost, 0) == 1])
    cost[np.sum(cost, 1) != 1] = False
    (row, col2) = np.where(cost)
    good_curr_idx = [ci for ri, ci in zip(row, col2) if ci in col1]
    good_prev_idx = [ri for ri, ci in zip(row, col2) if ci in good_curr_idx]
    return good_curr_idx, good_prev_idx
