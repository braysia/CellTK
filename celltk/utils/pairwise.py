from __future__ import division
import numpy as np
from scipy.spatial.distance import pdist


def one_to_one_assignment(binary_cost, value_cost):
    '''When mulitple True exists in row or column, it will pick one with the lowest value_cost.
    '''
    binary_cost = pick_closer(binary_cost, value_cost)
    binary_cost = pick_closer(binary_cost.T, value_cost.T)
    binary_cost = binary_cost.T
    return binary_cost


def one_to_two_assignment(binary_cost, value_cost):
    '''If there are more than two True in a row, make them to two.
    '''
    # First make sure the daughter is not shared by two parents
    binary_cost = pick_closer(binary_cost.T, value_cost.T)
    binary_cost = binary_cost.T
    # pick two based on value_cost
    binary_cost = pick_closer_two(binary_cost, value_cost)
    return binary_cost

def angle_assignment(binary_cost, dau_pts, par_pts, dot_thres, dist_thres, mass_cost, weight):
    '''
    Calculates angle between all possible daughter vector pairs
    Eliminates ones that are above the thres
    Assigns daughter pairs to parents based on the remaining using mass, angle, and distance info
    '''

    #iterate over each parent
    for x in xrange(0, binary_cost.shape[0]):

        #select the possible candidate daughters from the binary matrix and get x,y coordinates
        bin_row = binary_cost[x, :]
        dau_idx = np.transpose(np.nonzero(binary_cost[x, :]))
        cand_par_xy = par_pts[x]
        cand_dau_xy = [dau_pts[n] for n, i in enumerate(bin_row) if i]

        #zero matrix to be filled in later by selected daughter pairs
        binary_cost[x, :] = False

        #if greater than one daughter is matched to each parent
        if len(cand_dau_xy) > 1:

            #calculate all dot_products/distance fractions
            dot_prod, dist_error = pairwise_dot_distance(cand_par_xy, cand_dau_xy)

            #keep cells which meet both thresholds
            dau_pairs = np.transpose(np.nonzero((np.logical_and(dot_prod<=dot_thres, dist_error<=dist_thres))))

            #if only one possible daughter pair is found, fill in the values immediately
            #if greater than one pair is found, assign it based on a combined mass/angle cost
            if len(dau_pairs) == 1:
                binary_cost[x, :][dau_idx[dau_pairs[0][0]]] = True
                binary_cost[x, :][dau_idx[dau_pairs[0][1]]] = True
            elif len(dau_pairs) > 1:

                #calculate the cost for each daughter pair based on available information
                costs = []
                for d in dau_pairs:
                    i1 = dau_idx[d[0]][0]
                    i2 = dau_idx[d[1]][0]
                    mass_error = (1 - weight) * (np.abs(np.sum([mass_cost[x, i1], mass_cost[x, i2]])))
                    angle_error = weight * ((1 - np.abs(dot_prod[d[0], d[1]])) + dist_error[d[0], d[1]])
                    costs.append(mass_error + angle_error)

                #assign the lowest cost daughter
                min_dp = np.argmin(np.abs(costs))
                binary_cost[x, :][dau_idx[dau_pairs[min_dp][0]]] = True
                binary_cost[x, :][dau_idx[dau_pairs[min_dp][1]]] = True

    return binary_cost

def pairwise_dot_distance(par_xy, dau_xy):
    '''
    Returns 2 matrices that are len(daughter_cells)xlen(daughter_cells)
    dot contains the normalized dot product for the daughter cell pairs and the parent
    distance_error is how far the parent is from the midpoint of the line connecting the daughter cells
    '''
    
    #calculate vector from the parent cell to each daughter
    vectors =[]
    for (y, x) in dau_xy:
        vectors.append(np.array([par_xy[0] - y, par_xy[1] - x]))
    
    #calculate all normalized dot products of those vectors
    dot = np.ones((len(vectors), len(vectors)))
    for i in xrange(0, len(vectors)):
        for j in xrange(i, len(vectors)):
            if not i == j:
                dot[i, j] = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))

    #find line between daughter cells, find closest point from parent cell on that line, determine fractional distance
    distance_error = np.ones((len(dau_xy), len(dau_xy)))
    for i in xrange(0, len(dau_xy)):
        for j in xrange(i, len(dau_xy)):
            if not i == j:
                d1 = dau_xy[i]
                d2 = dau_xy[j]

                #screen to avoid divide by 0 errors
                if d1[0] == d2[0] or d1[1] == d2[1]:
                    distance_error[i, j] = 1 # if daughter points are the same, assign very high cost
                elif (d1[0] == par_xy[0] or d2[0] == par_xy[0]) or (d1[1] == par_xy[1] or d2[1] == par_xy[1]):
                    distance_error[i, j] = 1 # if parent point is in line with daughter, assign high cost
                else:
                    #determine slope of line between daughters and the perpendicular slope
                    daughter_slope = (d1[0] - d2[0]) / (d1[1] - d2[1])
                    reciprocal_slope = -1. / daughter_slope

                    #basic algebra to find closest point
                    closest_x = (daughter_slope * d2[1] - reciprocal_slope * par_xy[1] + par_xy[0] - d2[0]) / (daughter_slope - reciprocal_slope)
                    closest_y = reciprocal_slope * (closest_x - par_xy[1]) + par_xy[0]

                    # find where on the line the parent lies
                    total_dist = pdist([d1, d2])
                    parent_dist = pdist([d1, (closest_y, closest_x)])

                    distance_error[i, j] = np.abs(0.5 - parent_dist/total_dist)

    return dot, distance_error

def find_one_to_one_assign(binary_cost):
    cost = binary_cost.copy()
    (_, col1) = np.where([np.sum(cost, 0) == 1])
    cost[np.sum(cost, 1) != 1] = False
    (row, col2) = np.where(cost)
    good_row = [ci for ri, ci in zip(row, col2) if ci in col1]
    good_col = [ri for ri, ci in zip(row, col2) if ci in good_row]
    return good_row, good_col


def pick_closer(binary_cost, value_cost):
    '''If there are several True in a row of binary_cost,
    it will pick one with the lowest value_cost.
    If mulitple elements have the same value_cost, it will pick the first one.
    '''
    for x in range(binary_cost.shape[0]):
        binary_row = binary_cost[x, :]
        value_row = value_cost[x, :]
        if binary_row.any():
            min_value = np.min(value_row[binary_row])
            idx = np.where(value_row == min_value)[0][0]
            binary_row[0:idx] = False
            binary_row[idx+1:] = False
    return binary_cost


def pick_closer_two(binary_cost, value_cost, PICK=2):
    for x in range(binary_cost.shape[0]):
        binary_row = binary_cost[x, :]
        value_row = value_cost[x, :]
        if binary_row.sum() > 1:
            binary_row_copy = binary_row.copy()
            sorted_idx = np.argsort(value_row[binary_row])
            binary_row[:] = False
            for i in sorted_idx[:PICK]:
                idx = np.where(value_row == value_row[binary_row_copy][i])[0][0]
                binary_row[idx] = True
    return binary_cost
