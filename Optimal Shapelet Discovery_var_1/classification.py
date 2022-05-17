# Lorenzo Bonasera 2022

import numpy as np
import sktime.datatypes._panel._convert as conv
import matplotlib.pyplot as plt


def predictModel(A, x, b, d, labels, X_test, y_test):
    t_rows, t_cols = X_test.shape
    tmp = np.zeros([t_rows, 1], dtype=int)
    Y_predict = np.hstack((np.reshape(y_test, (t_rows, 1)), tmp))
    num_nodes = len(d)

    # split
    if num_nodes > 1:
        for i in range(t_rows):
            if np.linalg.norm(x[0] - np.dot(A[0], X_test[i, :]), ord=1) < b[0]:
                if np.linalg.norm(x[1] - np.dot(A[1], X_test[i, :]), ord=1) < b[1]:
                    Y_predict[i, 1] = labels[0]

                elif np.linalg.norm(x[1] - np.dot(A[1], X_test[i, :]), ord=1) >= b[1]:
                    Y_predict[i, 1] = labels[1]

            elif np.linalg.norm(x[0] - np.dot(A[0], X_test[i, :]), ord=1) >= b[0]:
                if np.linalg.norm(x[2] - np.dot(A[2], X_test[i, :]), ord=1) < b[2]:
                    Y_predict[i, 1] = labels[2]

                elif np.linalg.norm(x[2] - np.dot(A[2], X_test[i, :]), ord=1) >= b[2]:
                    Y_predict[i, 1] = labels[3]
    else:
        for i in range(t_rows):
            if np.linalg.norm(x[0] - np.dot(A[0], X_test[i, :]), ord=1) < b[0]:
                Y_predict[i, 1] = labels[0]
            else:
                Y_predict[i, 1] = labels[1]


    acc = 1 - sum(np.minimum(abs(Y_predict[:, 1] - Y_predict[:, 0]), 1)) / t_rows

    return acc


def plotShapelets(x, d):

    for idx, val in enumerate(d):
        if val > 0.5:
            exemplar = x[idx]
            plt.figure()
            plt.plot(np.arange(0, len(exemplar)), exemplar, color='r', label='Shapelet')
            plt.legend()
            plt.show()