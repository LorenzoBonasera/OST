# Lorenzo Bonasera 2022

from preprocessing import preprocessTrain, preprocessTest
from classification import predictModel, plotShapelets
from training import generateModel, retrieveSolution
from sktime.datasets import load_UCR_UEA_dataset
from wildboar.ensemble import ShapeletForestClassifier
import sktime.datatypes._panel._convert as conv
import numpy as np

if __name__ == '__main__':

    ### Load data
    np.random.seed(1)

    dataset = 'Coffee'
    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='Train', return_X_y=True)
    X_train, y_train, y_set, scaler = preprocessTrain(X_train, y_train)

    maxSize = 100000

    if maxSize < X_train.shape[0]:
        idx = np.random.randint(0, X_train.shape[0], size=maxSize)
        X_train = X_train[idx, :]
        y_train = y_train[idx]

    clf = ShapeletForestClassifier(n_jobs=-1, random_state=1, min_shapelet_size=0.138, max_shapelet_size=0.140)
    clf.fit(X_train, y_train)

    # Testing
    X_test, y_test = load_UCR_UEA_dataset(name=dataset, split='Test', return_X_y=True)
    X_test, y_test = preprocessTest(X_test, y_test, y_set, scaler)

    score = clf.score(X_test, y_test)
    print("Accuracy:", score)



