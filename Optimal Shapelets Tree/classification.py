# Lorenzo Bonasera 2022


import numpy as np
import matplotlib.pyplot as plt
from math import log2


def postProcess(b, l):
    branches = len(b)
    depth = int(log2(branches + 1))
    process = np.zeros(branches)

    for i in range(len(l)):
        if l[i] == 0:
            if process[(i + branches - 1)//2] != 0:
                process[(i + branches - 1)//2] = 2
            elif (i + branches) % 2 == 0:
                process[(i + branches - 1)//2] = -1
            else:
                process[(i + branches - 1)//2] = 1

    for d in range(depth-2):
        for i in range(branches):
            if process[i] == 2:
                if process[(i - 1)//2] != 0:
                    process[(i - 1)//2] = 2
                elif i % 2 == 0:
                    process[(i - 1)//2] = -1
                else:
                    process[(i - 1)//2] = 1
                process[i] = -2

    return process


def computeLabel(A, A_hat, b, branches, data, exemplar, node=0, process=None):
    H = len(exemplar) - len(A[node])
    if process[node] == -1:
        if 2 * node + 1 < branches:
            return computeLabel(A, A_hat, b, branches, data, exemplar, node * 2 + 1, process)
        return int(2 * node + 1 - branches)
    if process[node] == 1:
        if 2 * node + 1 < branches:
            return computeLabel(A, A_hat, b, branches, data, exemplar, node * 2 + 2, process)
        return int(2 * node + 2 - branches)

    idx = int(np.nonzero(A[node])[0])
    idx_hat = int(np.nonzero(A_hat[node])[0])
    distance = np.linalg.norm(exemplar[idx_hat:idx_hat+H] - data[idx:idx+H], ord=1)

    if 2*node + 1 < branches:
        if distance <= round(b[node], 3):
            return computeLabel(A, A_hat, b, branches, data, exemplar, node * 2 + 1, process)

        return computeLabel(A, A_hat, b, branches, data, exemplar, node * 2 + 2, process)

    if distance <= round(b[node], 3):
        return int(2*node + 1 - branches)

    return int(2*node + 2 - branches)


def predictModel(A, A_hat, b, l, labels, X_test, y_test, exemplar):
    t_rows, t_cols = X_test.shape
    tmp = np.zeros([t_rows, 1], dtype=int)
    Y_predict = np.hstack((np.reshape(y_test, (t_rows, 1)), tmp))
    branches = len(b)
    print("active leaves:", sum(l))
    process = postProcess(b, l)

    # split
    for i in range(t_rows):
        Y_predict[i, 1] = labels[computeLabel(A, A_hat, b, branches, X_test[i, :], exemplar, 0, process)]

    acc = 1 - sum(np.minimum(abs(Y_predict[:, 1] - Y_predict[:, 0]), 1)) / t_rows

    return acc


def plotShapelets(A_hat, d, exemplar):

    for idx, val in enumerate(d):
        if val > 0:
            idx_hat = int(np.nonzero(A_hat[idx])[0])
            H = len(exemplar) - len(A_hat[idx])
            plt.figure()
            plt.plot(np.arange(0, len(exemplar)), exemplar, color='b', label='Exemplar')
            plt.plot(np.arange(idx_hat, idx_hat+H), exemplar[idx_hat:idx_hat+H], color='r', label='Shapelet_{}'.format(idx))
            plt.legend()
            plt.show()