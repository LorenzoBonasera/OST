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


def compute_lb_ub(set_of_time_series, H):
    n = set_of_time_series.shape[0]
    J = set_of_time_series.shape[1]
    lb = np.zeros((n, H))
    ub = np.zeros((n, H))

    for i in range(n):
        for h in range(H):
            lb[i, h] = min(set_of_time_series[i][h:J-H+h])
            ub[i, h] = max(set_of_time_series[i][h:J - H + h])

    return lb, ub


def compute_big_M(lb, ub, lb_ex, ub_ex):
    n = lb.shape[0]
    H = lb.shape[1]
    bigM1 = np.zeros((n, H))
    bigM2 = np.zeros(n)

    for j in range(n):
        for h in range(H):
            bigM1[j, h] = max(ub_ex[0, h] - lb[j, h], ub[j, h] - lb_ex[0, h])
        bigM2[j] = sum(bigM1[j, :])

    return bigM1, bigM2


def findAncestors(A_L, A_R, child):
    if child == 0:
        return

    father = (child-1)//2
    if child % 2 != 0:
        A_L.append(father)
    else:
        A_R.append(father)

    findAncestors(A_L, A_R, father)


def generateModel(X_train, y_train, alpha, depth, H, epsilon, LT=1, exemplar=None, K=None, true_exemplar=None):

    # Parameters
    Nmin = ceil(0.05 * X_train.shape[0])
    #Nmin = 10
    J = X_train.shape[1]
    if K == None:
        K = len(set(y_train))

    # check depth
    if K > 2**depth:
        raise ValueError('Depth is too low to handle so many different classes!')

    # compute exemplar
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

    # compute big-Ms
    lb, ub = compute_lb_ub(X_train, H)
    lb_ex, ub_ex = compute_lb_ub(np.expand_dims(X_exemplar, axis=0), H)
    M1, M2 = compute_big_M(lb, ub, lb_ex, ub_ex)

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
    w = m.addVars(n, branch_nodes, vtype=GRB.BINARY, name="w")
    N_kt = m.addVars(K, leaf_nodes, vtype=GRB.INTEGER, name="N_kt")
    N_t = m.addVars(leaf_nodes, vtype=GRB.INTEGER, name="N_t")
    c_kt = m.addVars(K, leaf_nodes, vtype=GRB.BINARY, name="c")
    L = m.addVars(leaf_nodes, vtype=GRB.INTEGER, name="L")

    # New variables
    a = m.addVars(J-H, branch_nodes, vtype=GRB.BINARY, name='a')
    a_hat = m.addVars(J-H, branch_nodes, vtype=GRB.BINARY, name='a_hat')
    b = m.addVars(branch_nodes, vtype=GRB.CONTINUOUS, name="b")
    #d = m.addVars(branch_nodes, vtype=GRB.BINARY, name="d")

    # Auxiliary variable
    gamma = m.addVars(n, H, branch_nodes, vtype=GRB.BINARY, name='gamma')
    beta = m.addVars(n, H, branch_nodes, vtype=GRB.CONTINUOUS, name='beta')
    beta_2 = m.addVars(H, branch_nodes, vtype=GRB.CONTINUOUS, name='beta_2')
    s1 = m.addVars(n, H, branch_nodes, vtype=GRB.CONTINUOUS, name='s1')
    s2 = m.addVars(n, H, branch_nodes, vtype=GRB.CONTINUOUS, name='s2')

    # Objective function
    if leaf_nodes > K:
        m.setObjective(L.sum()/LT + alpha * l.sum(), GRB.MINIMIZE)
    else:
        m.setObjective(L.sum(), GRB.MINIMIZE)

    # Constraints
    for i in range(branch_nodes):
        # only one index
        m.addConstr(a.sum('*', i) == 1)
        m.addConstr(a_hat.sum('*', i) == 1)

        # upper and lower bounds of betas
        for j in range(n):
            for h in range(H):
                m.addConstr(beta[j, h, i] <= ub[j, h])
                m.addConstr(beta[j, h, i] >= lb[j, h])

        for h in range(H):
            m.addConstr(beta_2[h, i] <= ub_ex[0, h])
            m.addConstr(beta_2[h, i] >= lb_ex[0, h])

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
        A_L = []
        A_R = []
        findAncestors(A_L, A_R, i+branch_nodes)
        for j in range(n):
            m.addConstr(z[j, i] <= l[i])
            m.addConstr(quicksum((1 - w[j, t]) for t in A_L) + quicksum(w[j, t] for t in A_R) >= depth*z[j, i])

    for k in range(branch_nodes):
        for j in range(n):
            for h in range(H):
                m.addConstr(beta[j, h, k] == quicksum(a[p - h, k] * X_train[j, p] for p in range(h, J - H + h)))
                m.addConstr(beta_2[h, k] == quicksum(a_hat[p - h, k] * X_exemplar[p] for p in range(h, J - H + h)))
                m.addConstr(s2[j, h, k] - s1[j, h, k] == beta_2[h, k] - beta[j, h, k])

                m.addConstr(s1[j, h, k] <= M1[j, h] * gamma[j, h, k])
                m.addConstr(s2[j, h, k] <= M1[j, h] * (1 - gamma[j, h, k]))

            m.addConstr(sum(s1.select(j, '*', k)) + sum(s2.select(j, '*', k)) <= b[k] + M2[j] * w[j, k])
            m.addConstr(sum(s1.select(j, '*', k)) + sum(s2.select(j, '*', k)) >= b[k] + epsilon - M2[j] * (1 - w[j, k]))


    return m, branch_nodes, leaf_nodes, n, X_exemplar


def retrieveSolution(model, branch_nodes, leaf_nodes, J, H, n, K):
    m = model

    for v in m.getVars():
        if str(v.varName).startswith(('c')):
            print(v.varName, v.x)

    # Retrieve tree structure
    A = []
    A_hat = []
    b = []
    l = []
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
        A.append(temp)
        A_hat.append(temp_hat)

    for i in range(leaf_nodes):
        var = m.getVarByName('l' + '[' + str(i) + ']')
        l.append(int(var.x))

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
    for bt in b: print(bt)

    print('Elapsed time:', m.Runtime)

    return A, A_hat, b, l, labels





