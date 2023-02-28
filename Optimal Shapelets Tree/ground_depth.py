# Lorenzo Bonasera 2023

import matplotlib.pyplot as plt
from classification import predictModel, plotShapelets
from training import generateModel, retrieveSolution
from statistics import median
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
import numpy as np
from scipy.stats import norm
import time
import pandas as pd
import multiprocessing as mp
from scipy.stats import mode


def assignLabel(distance, b, depth, index, node=0, sign=1):
    if 2*node + 1 < 2**depth - 1:
        if distance[node][index] + 0.0002 <= b[node]:
            return assignLabel(distance, b, depth, index, node * 2 + 1, sign)

        return assignLabel(distance, b, depth, index, node * 2 + 2, -sign)

    if distance[node][index] + 0.0002 <= b[node]:
        return int((sign + 1)/2)

    return int((-sign + 1)/2)


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


def computeAll(depth_true):
    m = 1000
    J = 100
    params = []
    in_sample_mean = []
    in_sample_std = []
    out_sample_mean = []
    out_sample_std = []
    cross_leaves_mean = []
    cross_leaves_std = []
    cross_depth_mean = []
    cross_depth_std = []
    leaves_mean = []
    leaves_std = []

    epsilon = 0.0002

    H = 9
    branch_nodes_true = 2 ** depth_true - 1
    in_sample = []
    out_of_sample = []
    cross_validated_leaves = []
    cross_validated_depth = []
    leaves = []

    for t in range(5):
        A_hat_true = []
        A_true = []
        count = 1
        probs = np.random.binomial(1, 5 / 6, branch_nodes_true)

        for i in range(branch_nodes_true):
            if i == 0:
                mat = np.zeros((H, J))
                idx = np.random.choice(J - H + 1, 1).item()
                submat = np.eye(H, H)
                mat[:, idx:idx + H] = submat
                A_hat_true.append(mat)

                mat = np.zeros((H, J))
                idx = np.random.choice(J - H + 1, 1).item()
                submat = np.eye(H, H)
                mat[:, idx:idx + H] = submat
                A_true.append(mat)

                count += 1

            else:
                if probs[i] == 1 and probs[(i - 1) // 2] == 1:
                    mat = np.zeros((H, J))
                    idx = np.random.choice(J - H + 1, 1).item()
                    submat = np.eye(H, H)
                    mat[:, idx:idx + H] = submat
                    A_hat_true.append(mat)

                    mat = np.zeros((H, J))
                    idx = np.random.choice(J - H + 1, 1).item()
                    submat = np.eye(H, H)
                    mat[:, idx:idx + H] = submat
                    A_true.append(mat)

                    count += 1

                else:
                    mat = np.zeros((H, J))
                    A_hat_true.append(mat)
                    A_true.append(mat)

        leaves.append(count)

        # The Wiener process parameter.
        delta = 1
        # T = 1
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
            exemplar = X[ex, :]

            # cutoff values!
            b = []
            distance_branch = []
            for i in range(branch_nodes_true):
                dist = []
                for j in range(m):
                    distance = np.linalg.norm(np.dot(A_hat_true[i], exemplar) - np.dot(A_true[i], X[j, :]), ord=1)
                    dist.append(round(distance, 3))
                b.append(round(median(dist), 3))
                distance_branch.append(dist)

            #print(b)

            for i in range(m):
                y[i] = assignLabel(distance_branch, b, depth_true, i)

            print("ok")

            y = np.asarray(list(map(int, y)))
            y_set = set(y)

            embedding = np.arange(0, len(y_set))
            for idx, label in enumerate(y_set):
                for i, data in enumerate(y):
                    if data == label:
                        y[i] = embedding[idx]

            #print(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=1, shuffle=True,
                                                                stratify=y)

            ### Training phase
            kf = KFold(n_splits=5, shuffle=True, random_state=1)
            alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]
            depth_values = [1, 2, 3]
            means = np.zeros((len(depth_values), len(alpha_values)))

            for depth in depth_values:

                for j, alpha in enumerate(alpha_values):
                    scores_alpha = []

                    for train_index, test_index in kf.split(X_train):
                        X_train_cv, X_val = X_train[train_index, :], X_train[test_index, :]
                        y_train_cv, y_val = y_train[train_index], y_train[test_index]
                        baseline = np.sum(y_train_cv == mode(y_train_cv, keepdims=True))
                        model, branch_nodes, leaf_nodes, n, _ = generateModel(X_train_cv, y_train_cv, alpha, depth, H,
                                                                              epsilon, LT=baseline, K=2, true_exemplar=exemplar)

                        print("Solving MIP model for cv:")
                        model.setParam('Presolve', 2)
                        model.setParam('OutputFlag', 0)
                        model.setParam('MIPGapAbs', 1 / n)
                        model.setParam('TimeLimit', 60)
                        model.optimize()

                        if model.getAttr('SolCount') >= 1:
                            A, A_hat, b, d, labels = retrieveSolution(model, branch_nodes, leaf_nodes, J, H, n, 2)
                            score = predictModel(A, A_hat, b, d, labels, X_val, y_val, exemplar)
                            scores_alpha.append(score)
                        else:
                            scores_alpha.append(0)

                    means[depth - 1, j] = np.array(scores_alpha).mean()

                    if depth == 1:
                        break

            ind = np.unravel_index(np.argmax(means, axis=None), means.shape)
            best_depth = depth_values[ind[0]]
            best_alpha = alpha_values[ind[1]]
            baseline = np.sum(y_train == mode(y_train, keepdims=True))
            model, branch_nodes, leaf_nodes, n, _ = generateModel(X_train, y_train, best_alpha, best_depth, H, epsilon,
                                                                  LT=baseline, K=2, true_exemplar=exemplar)

            print("Solving MIP model:")
            model.setParam('Presolve', 2)
            model.setParam('TimeLimit', 600)
            model.setParam('MIPGapAbs', 1 / n)
            model.optimize()

            if model.getAttr('SolCount') >= 1:
                A, A_hat, b, d, labels = retrieveSolution(model, branch_nodes, leaf_nodes, J, H, n, 2)
                score = predictModel(A, A_hat, b, d, labels, X_test, y_test, exemplar)
                out_of_sample.append(score)
                score = predictModel(A, A_hat, b, d, labels, X_train, y_train, exemplar)
                in_sample.append(score)
                cross_validated_leaves.append(sum(d) + 1)
                cross_validated_depth.append(best_depth)

    params.append(depth_true)
    in_sample_mean.append(np.mean(in_sample))
    in_sample_std.append(np.std(in_sample))
    out_sample_mean.append(np.mean(out_of_sample))
    out_sample_std.append(np.std(out_of_sample))
    cross_depth_mean.append(np.array(cross_validated_depth).mean())
    cross_depth_std.append(np.array(cross_validated_depth).std())
    cross_leaves_mean.append(np.array(cross_validated_leaves).mean())
    cross_leaves_std.append(np.array(cross_validated_leaves).std())
    leaves_mean.append(np.array(leaves).mean())
    leaves_std.append(np.array(leaves).std())

    columns = ['True depth', 'In-sample mean', 'In-sample std', 'Out-of-sample mean', 'Out-of-sample std',
               'Mean leaves', 'Std leaves', 'CrossVal mean leaves', 'CrossVal std leaves', 'Max depth mean',
               'Max depth std']
    df = pd.DataFrame(columns=columns)
    df['True depth'] = np.array(params)
    df['In-sample mean'] = np.array(in_sample_mean)
    df['In-sample std'] = np.array(in_sample_std)
    df['Out-of-sample mean'] = np.array(out_sample_mean)
    df['Out-of-sample std'] = np.array(out_sample_std)
    df['Mean leaves'] = np.array(leaves_mean)
    df['Std leaves'] = np.array(leaves_std)
    df['CrossVal mean leaves'] = np.array(cross_leaves_mean)
    df['CrossVal std leaves'] = np.array(cross_leaves_std)
    df['Max depth mean'] = np.array(cross_depth_mean)
    df['Max depth std'] = np.array(cross_depth_std)

    return df


if __name__ == '__main__':

    np.random.seed(2)
    parameter_range = [1, 2, 3, 4]
    # remember to change H inside computeAll!
    pool = mp.Pool(4)

    list_of_df = pool.map(computeAll, [H_true for H_true in parameter_range])
    df = pd.concat(list_of_df)
    df.to_csv('results_H_9_baseline.csv', sep=';', index=False)






