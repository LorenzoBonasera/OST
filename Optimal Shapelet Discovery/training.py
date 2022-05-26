# Lorenzo Bonasera 2022

from gurobipy import *
import numpy as np


def generateModel(X, y, alpha, depth, H, epsilon, exemplar=None, maxSize=None):

    # Parameters
    if maxSize is None:
        maxSize = X.shape[0]
    Nmin = round(0.05 * maxSize)
    J = X.shape[1]
    K = len(set(y))

    # check depth
    if K > 2**depth:
        raise ValueError('Depth is too low to handle so many different classes!')

    # Compute big-M
    M3 = 1
    M1 = H * M3 + epsilon
    M2 = H * M3
    M4 = 2 * M3 + epsilon

    print("M3:", M3)
    print("M2/M1:", M1)

    X_train = X
    y_train = y
    
    if maxSize > X.shape[0]:
        raise ValueError('MaxSize is larger than dataset size!')

    if maxSize < X.shape[0]:
        idx = np.random.randint(0, X.shape[0], size=maxSize)
        X_train = X[idx, :]
        y_train = y_train[idx]

    ex = exemplar
    if ex is None:
        ex = np.random.randint(0, maxSize)
        print(ex)

    X_exemplar = X[ex, :]
    X_train = np.delete(X_train, ex, axis=0)
    y_train = np.delete(y_train, ex)
    n = X_train.shape[0]

    # Tree structure
    leaf_nodes = 2**depth
    branch_nodes = 2**depth - 1

    # Y matrix
    Y = np.zeros([n, K], dtype=int) - 1
    Y[range(n), y_train] = 1

    # print(Y)

    # Model
    m = Model('mip1')

    # Variables
    l = m.addVars(leaf_nodes, vtype=GRB.BINARY, name="l")
    z = m.addVars(n, leaf_nodes, vtype=GRB.BINARY, name="z")
    N_kt = m.addVars(K, leaf_nodes, vtype=GRB.INTEGER, name="N_kt")
    N_t = m.addVars(leaf_nodes, vtype=GRB.INTEGER, name="N_t")
    c_kt = m.addVars(K, leaf_nodes, vtype=GRB.BINARY, name="c")
    L = m.addVars(leaf_nodes, vtype=GRB.INTEGER, name="L")

    # New variables
    a = m.addVars(H, J, branch_nodes, vtype=GRB.BINARY, name='a')
    a_hat = m.addVars(H, J, branch_nodes, vtype=GRB.BINARY, name='a_hat')
    b = m.addVars(branch_nodes, vtype=GRB.CONTINUOUS, name="b")
    d = m.addVars(branch_nodes, vtype=GRB.BINARY, name="d")

    # Auxiliary variable
    theta = m.addVars(H, n, branch_nodes, vtype=GRB.CONTINUOUS, name='theta')
    gamma = m.addVars(H, n, branch_nodes, vtype=GRB.BINARY, name='gamma')

    # Objective function
    if K > 2 or depth == 1:
        m.setObjective(L.sum(), GRB.MINIMIZE)
    else:
        m.setObjective(L.sum() + alpha * d.sum(), GRB.MINIMIZE)

    # Constraints
    for i in range(branch_nodes):
        m.addConstr(b[i] <= M1 * d[i])
        if i > 0:
            m.addConstr(d[i] <= d[(i-1)//2])
        # enforce diagonal structure
        for h in range(H):
            m.addConstr(a.sum(h, '*', i) == d[i])
            m.addConstr(a_hat.sum(h, '*', i) == d[i])
        for h in range(H-1):
            for j in range(J-1):
                m.addConstr(a[h,j,i] == a[h+1,j+1,i])
                m.addConstr(a_hat[h,j,i] == a_hat[h+1,j+1,i])

        for h in range(H):
            for j in range(J):
                if j - h > J - H:
                    m.addConstr(a[h, j, i] == 0)
                    m.addConstr(a_hat[h, j, i] == 0)
                if j < h:
                    m.addConstr(a[h, j, i] == 0)
                    m.addConstr(a_hat[h, j, i] == 0)

        # theta big M
        for h in range(H):
            for j in range(n):
                m.addConstr(theta[h, j, i] <= M3 * d[i])

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
                    # ok cambiare qua con le theta
                    #sx
                    if idx + 1 <= length // 2:
                        for h in range(H):
                            m.addConstr(sum(a_hat.select(h, '*', k) * X_exemplar - a.select(h, '*', k) * X_train[j, :]) <=
                                        sum(a.select(h, '*', k) * X_train[j, :] - a_hat.select(h, '*',k) * X_exemplar) + M4*gamma[h, j, k] - epsilon*(1 - gamma[h, j, k]))
                            m.addConstr(sum(a_hat.select(h, '*', k) * X_exemplar - a.select(h, '*', k) * X_train[j, :]) >=
                                        sum(a.select(h, '*', k) * X_train[j, :] - a_hat.select(h, '*',k) * X_exemplar) - M4*(1 - gamma[h, j, k]))
                            m.addConstr(theta[h, j, k] <= sum(a.select(h, '*', k) * X_train[j, :] - a_hat.select(h, '*',k) * X_exemplar) + M4*gamma[h, j, k])
                            m.addConstr(theta[h, j, k] >= sum(a.select(h, '*', k) * X_train[j, :] - a_hat.select(h, '*', k) * X_exemplar) - M4*gamma[h, j, k])
                            m.addConstr(theta[h, j, k] <= sum(a_hat.select(h, '*', k) * X_exemplar - a.select(h, '*', k) * X_train[j, :]) + M4*(1 - gamma[h, j, k]))
                            m.addConstr(theta[h, j, k] >= sum(a_hat.select(h, '*', k) * X_exemplar - a.select(h, '*', k) * X_train[j, :]) - M4*(1 -gamma[h, j, k]))
                        m.addConstr(sum(theta.select('*', j, k)) + epsilon <= b[k] + M1 * (1 - z[j, i]))

                    # qui idem
                    #dx
                    elif idx + 1 > length // 2:
                        for h in range(H):
                            m.addConstr(sum(a_hat.select(h, '*', k) * X_exemplar - a.select(h, '*', k) * X_train[j, :]) <=
                                        sum(a.select(h, '*', k) * X_train[j, :] - a_hat.select(h, '*',k) * X_exemplar) + M4*gamma[h, j, k] - epsilon*(1 - gamma[h, j, k]))
                            m.addConstr(sum(a_hat.select(h, '*', k) * X_exemplar - a.select(h, '*', k) * X_train[j, :]) >=
                                        sum(a.select(h, '*', k) * X_train[j, :] - a_hat.select(h, '*',k) * X_exemplar) - M4*(1 - gamma[h, j, k]))
                            m.addConstr(theta[h, j, k] <= sum(a.select(h, '*', k) * X_train[j, :] - a_hat.select(h, '*',k) * X_exemplar) + M4*gamma[h, j, k])
                            m.addConstr(theta[h, j, k] >= sum(a.select(h, '*', k) * X_train[j, :] - a_hat.select(h, '*', k) * X_exemplar) - M4*gamma[h, j, k])
                            m.addConstr(theta[h, j, k] <= sum(a_hat.select(h, '*', k) * X_exemplar - a.select(h, '*', k) * X_train[j, :]) + M4*(1 - gamma[h, j, k]))
                            m.addConstr(theta[h, j, k] >= sum(a_hat.select(h, '*', k) * X_exemplar - a.select(h, '*', k) * X_train[j, :]) - M4*(1 - gamma[h, j, k]))
                        m.addConstr(sum(theta.select('*', j, k)) >= b[k] - M2 * (1 - z[j, i]))
                else:
                    continue

    #m.addConstr(L.sum() <= 3)
    return m, branch_nodes, leaf_nodes, n, X_exemplar


def retrieveSolution(model, branch_nodes, leaf_nodes, J, H, n, K):
    m = model

    for v in m.getVars():
        if str(v.varName).startswith(('l', 'L', 'N', 'd', 'c', 'b')):
            print(v.varName, v.x)

    # Retrieve tree structure
    A = []
    A_hat = []
    b = []
    d = []
    theta = []
    for i in range(branch_nodes):
        temp = np.zeros((H, J))
        temp_hat = np.zeros((H, J))
        temp_theta = np.zeros(n)
        for h in range(H):
            for j in range(J):
                var = m.getVarByName('a' + '[' + str(h) + ',' + str(j) + ',' + str(i) + ']')
                temp[h, j] = var.x
                var = m.getVarByName('a_hat' + '[' + str(h) + ',' + str(j) + ',' + str(i) + ']')
                temp_hat[h, j] = var.x

        for j in range(n):
            temp_theta_sum = np.zeros(H)
            for h in range(H):
                var = m.getVarByName('theta' + '[' + str(h) + ',' + str(j) + ',' + str(i) + ']')
                temp_theta_sum[h] = var.x
            temp_theta[j] = sum(temp_theta_sum)

        var = m.getVarByName('b' + '[' + str(i) + ']')
        b.append(var.x)
        var = m.getVarByName('d' + '[' + str(i) + ']')
        d.append(var.x)
        A.append(temp)
        A_hat.append(temp_hat)
        theta.append(temp_theta)

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

    print("Matrices A_t:")
    for a in A: print(a)
    print("Matrices hat(A_t):")
    for a in A_hat: print(a)
    print("Cut-off values b_t:")
    for bt in b: print(bt)

    print('Elapsed time:', m.Runtime)

    return A, A_hat, b, d, labels





