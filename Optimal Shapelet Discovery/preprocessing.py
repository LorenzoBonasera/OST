# Lorenzo Bonasera 2022

import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMinMax
import sktime.datatypes._panel._convert as conv
from sklearn.model_selection import train_test_split


def preprocessTrain(X, y, maxSize=None):
    if maxSize is None:
        maxSize = X.shape[0]

    elif maxSize > X.shape[0]:
        raise ValueError('MaxSize is larger that dataset size!')

    y = np.asarray(list(map(int, y)))
    y_set = set(y)

    embedding = np.arange(0, len(y_set))
    for idx, label in enumerate(y_set):
        for i, data in enumerate(y):
            if data == label:
                y[i] = embedding[idx]

    X = conv.from_nested_to_2d_array(X)
    X = X.to_numpy()

    # scale data
    scaler = TimeSeriesScalerMinMax()
    X = scaler.fit_transform(X)
    X = X[:, :, 0]
    #X = np.round(X, 3)

    if maxSize < X.shape[0]:
        X, _, y, _ = train_test_split(X, y, test_size=X.shape[0]-maxSize, stratify=y)

    return X, y, y_set, scaler


def preprocessTest(X, y, y_set, scaler):
    y = np.asarray(list(map(int, y)))

    embedding = np.arange(0, len(y_set))
    for idx, label in enumerate(y_set):
        for i, data in enumerate(y):
            if data == label:
                y[i] = embedding[idx]

    X = conv.from_nested_to_2d_array(X)
    X = X.to_numpy()

    # scale data
    X = scaler.fit_transform(X)
    X = X[:, :, 0]
    #X = np.round(X, 3)

    return X, y





