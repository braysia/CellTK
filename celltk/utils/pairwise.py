import numpy as np


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

def angle_assignment(binary_cost, vectors, thres, mass_cost, weight):
    '''
    Calculates angle between all possible daughter vector pairs
    Eliminates ones that are above the thres
    Assigns daughter pairs to parents based on the remaining 
    '''
    for x in xrange(0, binary_cost.shape[0]):
        bin_row = binary_cost[x, :]
        dau_idx = np.transpose(np.nonzero(binary_cost[x, :]))
        cand = vectors[x, :][bin_row==1]

        binary_cost[x, :] = 0
        if len(cand) > 1:
            dot_prod = pairwise_dotproduct(cand)
            mask = np.where(dot_prod<=thres, dot_prod, 0) 
            dau_pairs = np.transpose(np.nonzero(mask)) #returns coordinates of nonzero entries in mask

            if len(dau_pairs) == 1:
                binary_cost[x, :][dau_idx[dau_pairs[0][0]]] = 1
                binary_cost[x, :][dau_idx[dau_pairs[0][1]]] = 1
            elif len(dau_pairs) > 1: # greater than one pair found, fix based on combinde angle/mass
                costs = []
                for d in dau_pairs:
                    i1 = dau_idx[d[0]][0]
                    i2 = dau_idx[d[1]][0]
                    mass_error = (1 - weight) * (np.abs(np.sum([mass_cost[x, i1], mass_cost[x, i2]])))
                    angle_error = weight * (1 - np.abs(dot_prod[d[0], d[1]]))
                    costs.append(mass_error + angle_error)

                min_dp = np.argmin(np.abs(costs))
                binary_cost[x, :][dau_idx[dau_pairs[min_dp][0]]] = 1
                binary_cost[x, :][dau_idx[dau_pairs[min_dp][1]]] = 1

    return binary_cost


def pairwise_dotproduct(vectors):
    vec = np.empty((len(vectors), len(vectors)))

    for i in xrange(0, len(vectors)):
        for j in xrange(i, len(vectors)):
            vec[i, j] = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
    return vec 

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
