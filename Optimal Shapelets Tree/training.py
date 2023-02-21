# Lorenzo Bonasera 2023


from gurobipy import *
from math import ceil
import numpy as np


def euclidean_distance(a, b):
    return np.linalg.norm(a - b, ord=1)


def KMedoids(series, k=1):
    n = series.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist[i][j] = euclidean_distance(series[i], series[j])
            dist[j][i] = dist[i][j]

    initial_medoid = np.random.choice(n)
    medoid = initial_medoid
    old_medoid = -1
    while medoid != old_medoid:
        dists = [euclidean_distance(series[i], series[medoid]) for i in range(n)]
        new_medoid = np.argmin(dists)
        old_medoid = medoid
        medoid = new_medoid

    print("medoid:", medoid)
    return series[medoid]


def compute_big_M(time_series, set_of_time_series, H):
    bigM1 = np.ones(set_of_time_series.shape[0])
    bigM2 = np.ones(set_of_time_series.shape[0])
    bigM3 = np.ones(set_of_time_series.shape[0])
    N = time_series.shape[0]
    time_series_subsequences = [time_series[j:j+H] for j in range(N - H + 1)]

    for i in range(set_of_time_series.shape[0]):
        max_distance = 0
        max_diff = 0
        min_diff = np.inf
        set_of_time_series_subsequences = [set_of_time_series[i, j:j+H] for j in range(N - H + 1)]
        for subsequence1 in time_series_subsequences:
            for subsequence2 in set_of_time_series_subsequences:
                diff = np.abs(subsequence1 - subsequence2)
                dist = np.max(diff)
                max_distance = max(max_distance, dist)
                max_diff = max(max_diff, sum(diff))
                min_diff = min(min_diff, sum(diff))
        bigM3[i] = max_distance
        bigM2[i] = min_diff
        bigM1[i] = max_diff
    return bigM1, bigM2


def generateModel(X_train, y_train, alpha, depth, H, epsilon, LT=1, exemplar=None, K=None, true_exemplar=None):

    # Parameters
    Nmin = ceil(0.05 * X_train.shape[0])
    J = X_train.shape[1]
    if K == None:
        K = len(set(y_train))

    # check depth
    if K > 2**depth:
        raise ValueError('Depth is too low to handle so many different classes!')

    if depth > 3:
        raise ValueError('Depth is too high!')

    # Compute big-M
    M3 = 1

    if true_exemplar is None:
        ex = exemplar
        if ex is None:
            ex = np.random.randint(0, X_train.shape[0])
            print(ex)

        X_exemplar = X_train[ex, :]
        #X_train = np.delete(X_train, ex, axis=0)
        #y_train = np.delete(y_train, ex)
    else:
        X_exemplar = true_exemplar

    n = X_train.shape[0]
    M1, M2 = compute_big_M(X_exemplar, X_train, H)
    M1 = M1 * 1
    M2 = M1 - M2
    M2 = M2 * 1
    M1 += epsilon
    M = max(M1)

    # Tree structure
    leaf_nodes = 2**depth
    branch_nodes = 2**depth - 1

    # Y matrix
    Y = np.zeros([n, K], dtype=int) - 1
    Y[range(n), y_train] = 1

    # Model
    m = Model('Optimal Shapelets Tree')

    # Variables
    l = m.addVars(leaf_nodes, vtype=GRB.BINARY, name="l")
    z = m.addVars(n, leaf_nodes, vtype=GRB.BINARY, name="z")
    N_kt = m.addVars(K, leaf_nodes, vtype=GRB.INTEGER, name="N_kt")
    N_t = m.addVars(leaf_nodes, vtype=GRB.INTEGER, name="N_t")
    c_kt = m.addVars(K, leaf_nodes, vtype=GRB.BINARY, name="c")
    L = m.addVars(leaf_nodes, vtype=GRB.INTEGER, name="L")

    # New variables
    a = m.addVars(J-H, branch_nodes, vtype=GRB.BINARY, name='a')
    a_hat = m.addVars(J-H, branch_nodes, vtype=GRB.BINARY, name='a_hat')
    b = m.addVars(branch_nodes, vtype=GRB.CONTINUOUS, name="b")
    d = m.addVars(branch_nodes, vtype=GRB.BINARY, name="d")

    # Auxiliary variable
    gamma = m.addVars(n, H, branch_nodes, vtype=GRB.BINARY, name='gamma')
    beta = m.addVars(n, H, branch_nodes, vtype=GRB.CONTINUOUS, name='beta')
    beta_2 = m.addVars(H, branch_nodes, vtype=GRB.CONTINUOUS, name='beta_2')
    s1 = m.addVars(n, H, branch_nodes, vtype=GRB.CONTINUOUS, name='s1')
    s2 = m.addVars(n, H, branch_nodes, vtype=GRB.CONTINUOUS, name='s2')


    # Objective function
    #if leaf_nodes > K:
    #    m.setObjective(L.sum()/LT + alpha * d.sum(), GRB.MINIMIZE)
    #else:
    m.setObjective(L.sum(), GRB.MINIMIZE)
    m.addConstr(d.sum() == alpha)

    # Constraints
    for i in range(branch_nodes):
        m.addConstr(b[i] <= M * d[i])
        #m.addConstr(b[i] >= MM * d[i])
        if i > 0:
            m.addConstr(d[i] <= d[(i-1)//2])

        # enforce diagonal structure
        m.addConstr(a.sum('*', i) == d[i])
        m.addConstr(a_hat.sum('*', i) == d[i])

    # enforce one leaf for each class
    for k in range(K):
        m.addConstr(c_kt.sum(k, '*') >= 1)

    for i in range(leaf_nodes):
        m.addConstr(L[i] >= 0)
        m.addConstr(N_t[i] == z.sum('*', i))
        m.addConstr(l[i] == c_kt.sum('*', i))
        m.addConstr(z.sum('*', i) >= Nmin*l[i])
        for j in range(K):
            m.addConstr(L[i] >= N_t[i] - N_kt[j, i] - n * (1 - c_kt[j, i]))
            m.addConstr(L[i] <= N_t[i] - N_kt[j, i] + n * c_kt[j, i])
            m.addConstr(N_kt[j, i] == 1 / 2 * sum(z.select('*', i) * (Y[:, j] + 1)))

    for i in range(n):
        m.addConstr(z.sum(i, '*') == 1)

    for i in range(leaf_nodes):
        for j in range(n):
            m.addConstr(z[j, i] <= l[i])

    all_branch_nodes = list(reversed(range(branch_nodes)))
    depth_dict = {}
    for i in range(depth):
        depth_dict[i] = sorted(all_branch_nodes[-2 ** i:])
        for j in range(2 ** i):
            all_branch_nodes.pop()

    all_leaf_nodes = list(range(leaf_nodes))
    branch_dict = {}
    for i in range(branch_nodes):
        for k in range(depth):
            if i in depth_dict[k]:
                floor_len = len(depth_dict[k])
                step = 2 ** depth // floor_len
                sliced_leaf = [all_leaf_nodes[i:i + step] for i in range(0, 2 ** depth, step)]
                idx = depth_dict[k].index(i)
                branch_dict[i] = sliced_leaf[idx]
            else:
                continue

    for j in range(n):
        for i in range(leaf_nodes):
            for k in range(branch_nodes):
                if i in branch_dict[k]:
                    length = len(branch_dict[k])
                    idx = branch_dict[k].index(i)
                    #sx
                    if idx + 1 <= length // 2:
                        for h in range(H):
                            m.addConstr(beta[j, h, k] == quicksum(a[p - h, k] * X_train[j, p] for p in range(h, J - H + h)))
                            m.addConstr(beta_2[h, k] == quicksum(a_hat[p - h, k] * X_exemplar[p] for p in range(h, J - H + h)))
                            m.addConstr(s2[j, h, k] - s1[j, h, k] == beta_2[h, k] - beta[j, h, k])

                            m.addConstr(s1[j, h, k] <= M3 * gamma[j, h, k])
                            m.addConstr(s2[j, h, k] <= M3 * (1 - gamma[j, h, k]))

                        m.addConstr(sum(s1.select(j, '*', k)) + sum(s2.select(j, '*', k)) + epsilon <= b[k] + M1[j] * (1 - z[j, i]))

                    #dx
                    elif idx + 1 > length // 2:
                        for h in range(H):
                            m.addConstr(beta[j, h, k] == quicksum(a[p - h, k] * X_train[j, p] for p in range(h, J - H + h)))
                            m.addConstr(beta_2[h, k] == quicksum(a_hat[p - h, k] * X_exemplar[p] for p in range(h, J - H + h)))
                            m.addConstr(s2[j, h, k] - s1[j, h, k] == beta_2[h, k] - beta[j, h, k])

                            m.addConstr(s1[j, h, k] <= M3 * gamma[j, h, k])
                            m.addConstr(s2[j, h, k] <= M3 * (1 - gamma[j, h, k]))
                        m.addConstr(sum(s1.select(j, '*', k)) + sum(s2.select(j, '*', k)) >= b[k] - M2[j] * (1 - z[j, i]))
                else:
                    continue

    return m, branch_nodes, leaf_nodes, n, X_exemplar


def retrieveSolution(model, branch_nodes, leaf_nodes, J, H, n, K):
    m = model

    for v in m.getVars():
        if str(v.varName).startswith(('d')):
            print(v.varName, v.x)

    # Retrieve tree structure
    A = []
    A_hat = []
    b = []
    d = []
    for i in range(branch_nodes):
        temp = np.zeros(J-H)
        temp_hat = np.zeros(J-H)
        for h in range(J-H):
            var = m.getVarByName('a' + '[' + str(h) + ',' + str(i) + ']')
            if var is None:
                print("here")
            temp[h] = var.x
            var = m.getVarByName('a_hat' + '[' + str(h) + ',' + str(i) + ']')
            temp_hat[h] = var.x

        var = m.getVarByName('b' + '[' + str(i) + ']')
        b.append(var.x)
        var = m.getVarByName('d' + '[' + str(i) + ']')
        d.append(var.x)
        A.append(temp)
        A_hat.append(temp_hat)

    # Obtain the labels of leaf nodes
    labels = np.zeros(leaf_nodes, dtype=int) - 1
    coff_c = np.zeros([K, leaf_nodes], dtype=int)

    for i in range(K):
        for j in range(leaf_nodes):
            tmp3 = m.getVarByName('c' + '[' + str(i) + ',' + str(j) + ']')
            coff_c[i, j] = int(tmp3.x)

    k_idx, t_idx = np.where(coff_c == 1)
    # for i in range(leaf_nodes):
    for i in range(len(k_idx)):
        labels[t_idx[i]] = k_idx[i]

    print("Cut-off values b_t:")
    for bt in b: print(round(bt, 3))

    print('Elapsed time:', m.Runtime)

    return A, A_hat, b, d, labels





