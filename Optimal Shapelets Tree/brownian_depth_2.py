# Lorenzo Bonasera 2022
import random
from classification import predictModel, plotShapelets
from training import generateModel, retrieveSolution
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
import numpy as np
from scipy.stats import norm
import time
from statistics import median
import pandas as pd
import multiprocessing as mp


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


def computeAll(H_true):
    m = 1000
    J = 100
    epsilon = 0.0002
    depth = 2
    branch_nodes = 2 ** depth - 1
    params = []
    in_sample_mean = []
    in_sample_std = []
    out_sample_mean = []
    out_sample_std = []
    cross_H_mean = []
    cross_H_std = []
    cross_D_mean = []
    cross_D_std = []
    leaves_mean = []
    leaves_std = []
    in_sample = []
    out_of_sample = []
    cross_validated_H = []
    cross_validated_D = []
    leaves = []

    for t in range(5):
        A_hat_true = []
        A_true = []
        count = 1

        for i in range(branch_nodes):
            prob = np.random.binomial(1, 5 / 6)
            if i == 0:
                prob = 1

            if prob == 1:
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

                count += 1

            else:
                mat = np.zeros((H_true, J))
                A_hat_true.append(mat)
                A_true.append(mat)

        leaves.append(count)

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
            exemplar = X[ex, :]

            # cut-off values
            b = []
            distance_branch = []
            for i in range(branch_nodes):
                dist = []
                for j in range(m):
                    distance = np.linalg.norm(np.dot(A_hat_true[i], exemplar) - np.dot(A_true[i], X[j, :]), ord=1)
                    dist.append(round(distance, 3))
                b.append(round(median(dist), 3))
                distance_branch.append(dist)

            print(b)

            # change depending on depth!
            for i in range(m):
                y[i] = assignLabel(distance_branch, b, depth, i)

            print("ok")

            y = np.asarray(list(map(int, y)))

            # print(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=1, shuffle=True,
                                                                stratify=y)

            ### Training phase
            kf = KFold(n_splits=5, shuffle=True, random_state=1)
            alpha = 1
            H_values = [2, 4, 6, 8, 10, 12, 14, 16, 18]
            means = []

            for H in H_values:
                scores_cv = []

                for train_index, test_index in kf.split(X_train):
                    X_train_cv, X_val = X_train[train_index, :], X_train[test_index, :]
                    y_train_cv, y_val = y_train[train_index], y_train[test_index]
                    model, branch_nodes, leaf_nodes, n, _ = generateModel(X_train_cv, y_train_cv, alpha, depth, H,
                                                                          epsilon, K=2, true_exemplar=exemplar)

                    print("Solving MIP model for cv:")
                    model.setParam('Presolve', 2)
                    model.setParam('MIPFocus', 1)
                    model.setParam('TimeLimit', 60)
                    model.setParam('OutputFlag', 0)
                    model.optimize()

                    if model.getAttr('SolCount') >= 1:
                        A, A_hat, b, d, labels = retrieveSolution(model, branch_nodes, leaf_nodes, J, H, n, 2)
                        score = predictModel(A, A_hat, b, d, labels, X_val, y_val, exemplar)
                        scores_cv.append(score)
                    else:
                        scores_cv.append(0)

                mean = np.array(scores_cv).mean()
                means.append(mean)

            best_H = H_values[np.argmax(means)]
            model, branch_nodes, leaf_nodes, n, _ = generateModel(X_train, y_train, alpha, depth, best_H, epsilon, K=2,
                                                                  true_exemplar=exemplar)

            print("Solving MIP model:")
            model.setParam('Presolve', 2)
            model.setParam('MIPFocus', 3)
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', 600)
            model.optimize()

            if model.getAttr('SolCount') >= 1:
                A, A_hat, b, d, labels = retrieveSolution(model, branch_nodes, leaf_nodes, J, best_H, n, 2)
                score = predictModel(A, A_hat, b, d, labels, X_test, y_test, exemplar)
                out_of_sample.append(score)
                score = predictModel(A, A_hat, b, d, labels, X_train, y_train, exemplar)
                in_sample.append(score)
                cross_validated_H.append(best_H)
                cross_validated_D.append(sum(d) + 1)

    params.append(H_true)
    in_sample_mean.append(np.mean(in_sample))
    in_sample_std.append(np.std(in_sample))
    out_sample_mean.append(np.mean(out_of_sample))
    out_sample_std.append(np.std(out_of_sample))
    cross_H_mean.append(np.array(cross_validated_H).mean())
    cross_H_std.append(np.array(cross_validated_H).std())
    cross_D_mean.append(np.array(cross_validated_D).mean())
    cross_D_std.append(np.array(cross_validated_D).std())
    leaves_mean.append(np.array(leaves).mean())
    leaves_std.append(np.array(leaves).std())

    columns = ['True H', 'In-sample mean', 'In-sample std', 'Out-of-sample mean', 'Out-of-sample std',
               'CrossVal mean H', 'CrossVal std H', 'Mean leaves', 'Std leaves', 'CrossVal mean leaves',
               'CrossVal std leaves']
    df = pd.DataFrame(columns=columns)
    df['True H'] = np.array(params)
    df['In-sample mean'] = np.array(in_sample_mean)
    df['In-sample std'] = np.array(in_sample_std)
    df['Out-of-sample mean'] = np.array(out_sample_mean)
    df['Out-of-sample std'] = np.array(out_sample_std)
    df['CrossVal mean H'] = np.array(cross_H_mean)
    df['CrossVal std H'] = np.array(cross_H_std)
    df['Mean leaves'] = np.array(leaves_mean)
    df['Std leaves'] = np.array(leaves_std)
    df['CrossVal mean leaves'] = np.array(cross_D_mean)
    df['CrossVal std leaves'] = np.array(cross_D_std)
    return df


if __name__ == '__main__':

    np.random.seed(2)
    #parameter_range = [3, 6, 9, 12, 15, 18]
    parameter_range = [9, 12, 15, 18]
    pool = mp.Pool(4)

    list_of_df = pool.map(computeAll, parameter_range)
    df = pd.concat(list_of_df)
    df.to_csv('results_depth_2_parallel.csv', sep=',', index=False)






