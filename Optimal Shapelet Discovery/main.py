# Lorenzo Bonasera 2022

from preprocessing import preprocessTrain, preprocessTest
from classification import predictModel, plotShapelets
from training import generateModel, retrieveSolution
from sktime.datasets import load_UCR_UEA_dataset
import numpy as np
import time

if __name__ == '__main__':
    # np.random.seed(2)

    ### Load data
    dataset = 'Wafer'
    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='Train', return_X_y=True)

    # Preprocessing
    maxSize = 101
    X_train, y_train, y_set, scaler = preprocessTrain(X_train, y_train, maxSize=maxSize)

    ### Parameters
    ex = 9
    J = X_train.shape[1]
    K = len(y_set)
    epsilon = 0.0001
    alpha = 1
    depth = 1
    H = 6

    ### Training phase
    model, branch_nodes, leaf_nodes, n, exemplar = generateModel(X_train, y_train, alpha, depth, H, epsilon, exemplar=ex)

    print("Solving MIP model:")
    #model.setParam('Heuristics', 0.5)
    model.setParam('PreSparsify', 1)
    #model.setParam('MIPFocus', 3)
    model.setParam('Presolve', 2)
    #model.setParam('TimeLimit', 600)
    model.optimize()

    A, A_hat, b, d, labels = retrieveSolution(model, branch_nodes, leaf_nodes, J, H, n, K)

    ### Test phase
    X_test, y_test = load_UCR_UEA_dataset(name=dataset, split='Test', return_X_y=True)
    X_test, y_test = preprocessTest(X_test, y_test, y_set, scaler)
    score = predictModel(A, A_hat, b, d, labels, X_test, y_test, exemplar)

    print("Accuracy:", score)

    #plotShapelets(A_hat, d, exemplar)





