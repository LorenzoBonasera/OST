# Lorenzo Bonasera 2022

import numpy as np
import matplotlib.pyplot as plt


def computeLabel(A, A_hat, b, branches, data, exemplar, node=0):
    distance = round(np.linalg.norm(np.dot(A_hat[node], exemplar) - np.dot(A[node], data), ord=1), 3)
    epsilon = 1e-4
    if 2*node + 1 < branches:
        if distance + epsilon <= round(b[node], 3):
            return computeLabel(A, A_hat, b, branches, data, exemplar, node * 2 + 1)

        return computeLabel(A, A_hat, b, branches, data, exemplar, node * 2 + 2)

    if distance + epsilon <= round(b[node], 3):
        return int(2*node + 1 - branches)

    return int(2*node + 2 - branches)


def predictModel(A, A_hat, b, d, labels, X_test, y_test, exemplar):
    t_rows, t_cols = X_test.shape
    tmp = np.zeros([t_rows, 1], dtype=int)
    Y_predict = np.hstack((np.reshape(y_test, (t_rows, 1)), tmp))
    branches = len(d)
    print("branches:", sum(d))

    # split
    for i in range(t_rows):
        Y_predict[i, 1] = labels[computeLabel(A, A_hat, b, branches, X_test[i, :], exemplar)]

    acc = 1 - sum(np.minimum(abs(Y_predict[:, 1] - Y_predict[:, 0]), 1)) / t_rows

    return acc


def plotShapelets(A_hat, d, exemplar):

    for idx, val in enumerate(d):
        if val > 0.5:
            mat = A_hat[idx]
            start = mat[0, :].argmax(axis=0)
            shapelet = np.dot(mat, exemplar)
            plt.figure()
            plt.plot(np.arange(0, len(exemplar)), exemplar, color='b', label='Exemplar')
            plt.plot(np.arange(start, start+len(shapelet)), shapelet, color='r', label='Shapelet_{}'.format(idx))
            plt.legend()
            plt.show()