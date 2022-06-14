# Lorenzo Bonasera 2022

import numpy as np
import sktime.datatypes._panel._convert as conv
import matplotlib.pyplot as plt


def predictModel(A, A_hat, b, d, labels, X_test, y_test, exemplar):
    t_rows, t_cols = X_test.shape
    tmp = np.zeros([t_rows, 1], dtype=int)
    Y_predict = np.hstack((np.reshape(y_test, (t_rows, 1)), tmp))
    num_splits = len(d)

    # split
    if num_splits == 7:
        for i in range(t_rows):
            if np.linalg.norm(np.dot(A_hat[0], exemplar) - np.dot(A[0], X_test[i, :]), ord=1) < b[0]:
                if np.linalg.norm(np.dot(A_hat[1], exemplar) - np.dot(A[1], X_test[i, :]), ord=1) < b[1]:
                    if np.linalg.norm(np.dot(A_hat[3], exemplar) - np.dot(A[3], X_test[i, :]), ord=1) < b[3]:
                        Y_predict[i, 1] = labels[0]

                    else:
                        Y_predict[i, 1] = labels[1]

                elif np.linalg.norm(np.dot(A_hat[4], exemplar) - np.dot(A[4], X_test[i, :]), ord=1) < b[4]:
                    Y_predict[i, 1] = labels[2]

                else:
                    Y_predict[i, 1] = labels[3]

            elif np.linalg.norm(np.dot(A_hat[2], exemplar) - np.dot(A[2], X_test[i, :]), ord=1) < b[2]:
                if np.linalg.norm(np.dot(A_hat[5], exemplar) - np.dot(A[5], X_test[i, :]), ord=1) < b[5]:
                    Y_predict[i, 1] = labels[4]

                else:
                    Y_predict[i, 1] = labels[5]

            elif np.linalg.norm(np.dot(A_hat[6], exemplar) - np.dot(A[6], X_test[i, :]), ord=1) < b[6]:
                Y_predict[i, 1] = labels[6]

            else:
                Y_predict[i, 1] = labels[7]

    elif num_splits == 3:
        for i in range(t_rows):
            if np.linalg.norm(np.dot(A_hat[0], exemplar) - np.dot(A[0], X_test[i, :]), ord=1) < b[0]:
                if np.linalg.norm(np.dot(A_hat[1], exemplar) - np.dot(A[1], X_test[i, :]), ord=1) < b[1]:
                    Y_predict[i, 1] = labels[0]

                else:
                    Y_predict[i, 1] = labels[1]

            elif np.linalg.norm(np.dot(A_hat[2], exemplar) - np.dot(A[2], X_test[i, :]), ord=1) < b[2]:
                Y_predict[i, 1] = labels[2]

            else:
                Y_predict[i, 1] = labels[3]

    else:
        for i in range(t_rows):
            if np.linalg.norm(np.dot(A_hat[0], exemplar) - np.dot(A[0], X_test[i, :]), ord=1) < b[0]:
                Y_predict[i, 1] = labels[0]

            else:
                Y_predict[i, 1] = labels[1]


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
