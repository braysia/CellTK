from __future__ import division
import numpy as np
from munkres import munkres


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


def prepare_costmat(cost, costDie, costBorn):
    '''d is cost matrix,
    often distance matrix where rows are previous and columns as current
    d contains NaN where tracking of those two objects are not possible.
    costDie and costBorn'''
    cost[np.isnan(cost)] = np.Inf  # give a large cost.
    costDieMat = np.float64(np.diag([costDie]*cost.shape[0]))  # diagonal
    costBornMat = np.float64(np.diag([costBorn]*cost.shape[1]))
    costDieMat[costDieMat == 0] = np.Inf
    costBornMat[costBornMat == 0] = np.Inf

    costMat = np.ones((sum(cost.shape), sum(cost.shape)))*np.Inf
    costMat[0:cost.shape[0], 0:cost.shape[1]] = cost
    costMat[-cost.shape[1]:, 0:cost.shape[1]] = costBornMat
    costMat[0:cost.shape[0], -cost.shape[0]:] = costDieMat
    lowerRightBlock = cost.transpose()
    costMat[cost.shape[0]:, cost.shape[1]:] = lowerRightBlock
    return costMat


def call_lap(cost, costDie, costBorn):
    costMat = prepare_costmat(cost, costDie, costBorn)
    t = munkres(costMat)
    topleft = t[0:cost.shape[0], 0:cost.shape[1]]
    return topleft


def pick_closer_binarycostmat(binarymat, distmat):
    '''
    pick closer cells if there are two similar nucleus within area
    '''
    twonuc = np.where(np.sum(binarymat, 1) == 2)[0]
    for ti in twonuc:
        di = distmat[ti, :]
        bi = binarymat[ti, :]
        binarymat[ti, :] = min(di[bi]) == di
    return binarymat


def pick_closer_cost(binarymat, distmat):
    '''
    pick closer cells if there are two similar nucleus within area
    '''
    twonuc = np.where(np.sum(binarymat, 1) > 1)[0]
    for ti in twonuc:
        di = distmat[ti, :]
        bi = binarymat[ti, :]
        binarymat[ti, :] = min(di[bi]) == di
        # bmat[ti, np.where(min(di[bi]) == di)[0][0]] = True
    return binarymat
