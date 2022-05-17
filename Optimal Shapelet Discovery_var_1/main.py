# Lorenzo Bonasera 2022

from preprocessing import preprocessTrain, preprocessTest
from classification import predictModel, plotShapelets
from training import generateModel, retrieveSolution
from sktime.datasets import load_UCR_UEA_dataset


if __name__ == '__main__':

    ### Load data

    dataset = 'TwoPatterns'
    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='Train', return_X_y=True)
    X_train, y_train, y_set, scaler = preprocessTrain(X_train, y_train)

    maxSize = 101
    J = X_train.shape[1]
    K = len(y_set)
    epsilon = 0.0005
    alpha = 1
    depth = 2
    H = 10

    ### Training phase

    model, branch_nodes, leaf_nodes, n = generateModel(X_train, y_train, alpha, depth, H, epsilon, maxSize=maxSize)

    print("Solving MIP model:")
    model.setParam('MIPFocus', 0)
    model.setParam('TimeLimit', 600)
    model.optimize()

    A, x, b, d, labels = retrieveSolution(model, branch_nodes, leaf_nodes, J, H, n, K)

    ### Test phase

    X_test, y_test = load_UCR_UEA_dataset(name=dataset, split='Test', return_X_y=True)
    X_test, y_test = preprocessTest(X_test, y_test, y_set, scaler)
    score = predictModel(A, x, b, d, labels, X_test, y_test)

    print("Accuracy:", score)

    plotShapelets(x, d)



