# Lorenzo Bonasera 2023


import tslearn.barycenters
from preprocessing import preprocessTrain, preprocessTest
from classification import predictModel, plotShapelets
from training import generateModel, retrieveSolution, KMedoids
from sktime.datasets import load_UCR_UEA_dataset
import numpy as np
from scipy.stats import mode


if __name__ == '__main__':
    np.random.seed(1)

    ### Load data
    dataset = 'ItalyPowerDemand'
    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='Train', return_X_y=True)

    # Preprocessing
    maxSize = None
    maxLength = None
    X_train, y_train, y_set, y_emb, scaler = preprocessTrain(X_train, y_train, maxSize=maxSize, maxLength=maxLength)

    ### Parameters
    K = len(y_emb)
    exemplars = []
    for y_type in y_emb:
        series = np.array([X_train[i] for i in range(X_train.shape[0]) if y_train[i] == y_type])
        ex = KMedoids(series)
        exemplars.append(ex)
        #exemplars.append(tslearn.barycenters.euclidean_barycenter(series).flatten())

    ex = 1
    J = X_train.shape[1]
    epsilon = 1e-5
    alpha = 1
    depth = 1
    H = 4
    baseline = np.sum(y_train == mode(y_train, keepdims=True))
    print("Baseline acc:", baseline)

    ### Training phase
    model, branch_nodes, leaf_nodes, n, exemplar = generateModel(X_train, y_train, alpha, depth, H, epsilon, true_exemplar=exemplars[ex], LT=baseline)
    print("Solving MIP model:")
    #model.setParam('Presolve', 2)
    model.setParam('TimeLimit', 600)
    model.setParam('FeasibilityTol', epsilon)
    model.optimize()

    A, A_hat, b, d, labels = retrieveSolution(model, branch_nodes, leaf_nodes, J, H, n, K)

    ### Test phase
    X_test, y_test = load_UCR_UEA_dataset(name=dataset, split='Test', return_X_y=True)
    X_test, y_test = preprocessTest(X_test, y_test, y_set, scaler, maxLength)
    score = predictModel(A, A_hat, b, d, labels, X_test, y_test, exemplar)
    print("Accuracy:", score)

    #plotShapelets(A_hat, d, exemplar)




