# Lorenzo Bonasera 2022


import numpy as np
import matplotlib.pyplot as plt


def computeLabel(A, A_hat, b, branches, data, exemplar, node=0, theta=None):
    H = len(exemplar) - len(A[node])
    if sum(A[node]) < 1:
        distance = 0
    else:
        idx = int(np.nonzero(A[node])[0])
        idx_hat = int(np.nonzero(A_hat[node])[0])
        distance = np.linalg.norm(exemplar[idx_hat:idx_hat+H] - data[idx:idx+H], ord=1)
    epsilon = 1e-4
    if 2*node + 1 < branches:
        if distance + epsilon <= round(b[node], 3):
            return computeLabel(A, A_hat, b, branches, data, exemplar, node * 2 + 1, theta)

        return computeLabel(A, A_hat, b, branches, data, exemplar, node * 2 + 2, theta)

    if distance + epsilon <= round(b[node], 3):
        return int(2*node + 1 - branches)

    return int(2*node + 2 - branches)


def predictModel(A, A_hat, b, d, labels, X_test, y_test, exemplar, theta=None):
    t_rows, t_cols = X_test.shape
    tmp = np.zeros([t_rows, 1], dtype=int)
    Y_predict = np.hstack((np.reshape(y_test, (t_rows, 1)), tmp))
    branches = len(d)
    print("active branches:", sum(d))

    # split
    for i in range(t_rows):
        if theta is not None:
            Y_predict[i, 1] = labels[computeLabel(A, A_hat, b, branches, X_test[i, :], exemplar, 0, theta[i])]
        else:
            Y_predict[i, 1] = labels[computeLabel(A, A_hat, b, branches, X_test[i, :], exemplar, 0)]

    acc = 1 - sum(np.minimum(abs(Y_predict[:, 1] - Y_predict[:, 0]), 1)) / t_rows

    return acc


def plotShapelets(A_hat, d, exemplar):

    for idx, val in enumerate(d):
        if val > 0.5:
            idx_hat = int(np.nonzero(A_hat[idx])[0])
            H = len(exemplar) - len(A_hat[idx])
            plt.figure()
            plt.plot(np.arange(0, len(exemplar)), exemplar, color='b', label='Exemplar')
            plt.plot(np.arange(idx_hat, idx_hat+H), exemplar[idx_hat:idx_hat+H], color='r', label='Shapelet_{}'.format(idx))
            plt.legend()
            plt.show()