# Lorenzo Bonasera 2022

import matplotlib.pyplot as plt

from classification import predictModel
from training import generateModel, retrieveSolution, KMedoids
from statistics import median
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
import numpy as np
from scipy.stats import norm
from scipy.stats import mode
import time
import pandas as pd
import multiprocessing as mp


def assignLabel(distance, b, depth, index, node=0, sign=1):
    if 2 * node + 1 < 2 ** depth - 1:
        if distance[node][index] + 0.0001 <= b[node]:
            return assignLabel(distance, b, depth, index, node * 2 + 1, sign)

        return assignLabel(distance, b, depth, index, node * 2 + 2, -sign)

    if distance[node][index] + 0.0001 <= b[node]:
        return int((sign + 1) / 2)

    return int((-sign + 1) / 2)


def brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta * np.sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def computeAll(length_true):
    size_true = 100
    m = 1000
    # J = 100
    J = length_true
    params = []
    in_sample_mean = []
    in_sample_std = []
    out_sample_mean = []
    out_sample_std = []

    epsilon = 0.0001

    depth = 1
    alpha = 1
    H_true = int(J / 5)
    branch_nodes_true = 2 ** depth - 1

    in_sample = []
    out_of_sample = []

    for t in range(5):
        A_hat_true = []
        A_true = []
        probs = np.random.binomial(1, 5 / 6, branch_nodes_true)

        for i in range(branch_nodes_true):
            if i == 0:
                mat = np.zeros((H_true, J))
                idx = np.random.choice(J - H_true + 1, 1).item()
                submat = np.eye(H_true, H_true)
                mat[:, idx:idx + H_true] = submat
                A_hat_true.append(mat)

                mat = np.zeros((H_true, J))
                idx = np.random.choice(J - H_true + 1, 1).item()
                submat = np.eye(H_true, H_true)
                mat[:, idx:idx + H_true] = submat
                A_true.append(mat)

            else:
                if probs[i] == 1 and probs[(i - 1) // 2] == 1:
                    mat = np.zeros((H_true, J))
                    idx = np.random.choice(J - H_true + 1, 1).item()
                    submat = np.eye(H_true, H_true)
                    mat[:, idx:idx + H_true] = submat
                    A_hat_true.append(mat)

                    mat = np.zeros((H_true, J))
                    idx = np.random.choice(J - H_true + 1, 1).item()
                    submat = np.eye(H_true, H_true)
                    mat[:, idx:idx + H_true] = submat
                    A_true.append(mat)

                else:
                    mat = np.zeros((H_true, J))
                    A_hat_true.append(mat)
                    A_true.append(mat)

        # The Wiener process parameter.
        delta = 1
        N = J - 1
        dt = 1

        for t_b in range(5):
            x = np.empty((m, N + 1))
            x[:, 0] = np.random.uniform(0, 1, m)

            brownian(x[:, 0], N, dt, delta, out=x[:, 1:])

            X = np.round(x, 3)
            scaler = TimeSeriesScalerMinMax()
            X = scaler.fit_transform(X)
            X = X[:, :, 0]

            ### Label
            y = np.ones(m)
            ex = np.random.choice(m, 1).item()
            true_exemplar = X[ex, :]

            # cut-off values
            b = []
            distance_branch = []
            for i in range(branch_nodes_true):
                dist = []
                for j in range(m):
                    distance = np.linalg.norm(np.dot(A_hat_true[i], true_exemplar) - np.dot(A_true[i], X[j, :]), ord=1)
                    dist.append(round(distance, 3))
                b.append(round(median(dist), 3))
                distance_branch.append(dist)

            # change depending on depth!
            for i in range(m):
                y[i] = assignLabel(distance_branch, b, depth, i)

            print("ok")

            y = np.asarray(list(map(int, y)))

            # print(y)
            test_size = 1 - size_true / m
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)
            ### we don't neel cross-validation here

            '''
            exemplars = []
            for y_type in y_train:
                series = np.array([X_train[i] for i in range(X_train.shape[0]) if y_train[i] == y_type])
                ex = KMedoids(series)
                exemplars.append(ex)

            ex = 0
            '''

            model, branch_nodes, leaf_nodes, n, exemplar = generateModel(X_train, y_train, alpha, 1, H_true, epsilon, K=2, true_exemplar=true_exemplar)


            print("Solving MIP model:")
            model.setParam('Presolve', 2)
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', 3600)
            model.optimize()

            if model.getAttr('SolCount') >= 1:
                A, A_hat, b, d, labels = retrieveSolution(model, branch_nodes, leaf_nodes, J, H_true, n, 2)
                score = predictModel(A, A_hat, b, d, labels, X_test, y_test, exemplar)
                out_of_sample.append(score)
                score = predictModel(A, A_hat, b, d, labels, X_train, y_train, exemplar)
                in_sample.append(score)


    params.append(length_true)
    in_sample_mean.append(np.mean(in_sample))
    in_sample_std.append(np.std(in_sample))
    out_sample_mean.append(np.mean(out_of_sample))
    out_sample_std.append(np.std(out_of_sample))

    columns = ['Training size', 'In-sample mean', 'In-sample std', 'Out-of-sample mean', 'Out-of-sample std']
    df = pd.DataFrame(columns=columns)
    df['Training size'] = np.array(params)
    df['In-sample mean'] = np.array(in_sample_mean)
    df['In-sample std'] = np.array(in_sample_std)
    df['Out-of-sample mean'] = np.array(out_sample_mean)
    df['Out-of-sample std'] = np.array(out_sample_std)
    return df.round(4)

if __name__ == '__main__':
    # this is for depth = 1!
    np.random.seed(2)
    parameter_range = [50, 100, 200, 400]
    pool = mp.Pool(4)

    list_of_df = pool.map(computeAll, parameter_range)
    df = pd.concat(list_of_df)

    df.to_csv('results_learning_curve_true_exemplar.csv', sep=';', index=False)
