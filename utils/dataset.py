import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import random

def read_dataset(filepath, scaler_lower_bound=0.1, scaler_upper_bound=1.1, test_size=0.2, debug=False):
    """
    Read the dataset from the specified file, automatically infer the mu_num corresponding
    to the dataset and perform the training/test set partitioning.
    :param filepath: dataset path
    :param scaler_lower_bound: scaling lower bound
    :param scaler_upper_bound: scaling upper bound
    :param test_size: testset ratio
    :param debug: debug for message print
    :return: scaled X_train, scaled X_test, Y_train for classification task, Y_test for classification task,
            Y_train for regression task, Y_test for regression task
    """
    if debug:
        print("[read_dataset] Reading dataset from", filepath)
    data = pd.read_csv(filepath)
    data_array = np.array(data)
    mu_num = int((data_array.shape[1] - 1) / 7)

    X = data_array[:, 0:-(mu_num + 1)]
    Y = np.atleast_2d(data_array[:, -(mu_num + 1):])

    scaler = MinMaxScaler(feature_range=(scaler_lower_bound, scaler_upper_bound))
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=test_size)
    Y_train_class = np.atleast_2d(Y_train[:, -(mu_num + 1)]).T
    Y_test_class = np.atleast_2d(Y_test[:, -(mu_num + 1)]).T
    Y_train_reg = np.atleast_2d(Y_train[:, -mu_num:])
    Y_test_reg = np.atleast_2d(Y_test[:, -mu_num:])

    if debug:
        print("[read_dataset] Read finished, mu_num={}, sample num={}, return.".format(mu_num, X.shape[0]))

    return X_train, X_test, Y_train_class, Y_train_reg, Y_test_class, Y_test_reg


def split_dataset2subsets_train(filepath, scaler_lower_bound=0.0, scaler_upper_bound=1.0,
                                subset_num=3, test_size=10000):
    """
    Read the dataset and sort it according to the mean of the square sum of the sample features.
    The last test_size samples are taken as the testset, and the rest are taken as the training set.
    The training set is sampled by bootstrapping to generate subset_num subsets.
    :param filepath: dataset path
    :param scaler_lower_bound: scaling lower bound
    :param scaler_upper_bound: scaling upper bound
    :param subset_num: number of subset
    :return: training features of each subset, training labels of each subset,
             training features for the backbone/MTFNN, training labels for the backbone/MTFNN,
             raw test set features
    """
    data = pd.read_csv(filepath)
    data_array = np.array(data)
    mu_num = int((data_array.shape[1] - 1) / 7)

    X = data_array[:, 0:-(mu_num + 1)]
    scaler = MinMaxScaler(feature_range=(scaler_lower_bound, scaler_upper_bound))
    X_scaled = scaler.fit_transform(X)
    Y = np.atleast_2d(data_array[:, -(mu_num + 1):])

    mu_vector_square = np.zeros((X_scaled.shape[0], mu_num))
    for i in range(X_scaled.shape[0]):
        for j in range(mu_num):
            mu_vector_square[i][j] = np.sum(X_scaled[i, j * 6:(j + 1) * 6] ** 2)
    global_mu_vector = np.array([np.sum(mu_vector_square[i, :]) / mu_num for i in range(mu_vector_square.shape[0])])
    sorted_global_mu_vector_indices = np.argsort(global_mu_vector)

    Xs = []
    Ys = []
    available_indicies = [sorted_global_mu_vector_indices[i] for i in range(X_scaled.shape[0] - test_size)]
    for i in range(subset_num - 1):
        rd_indicies = [random.choice(available_indicies) for _ in range(len(available_indicies))]
        Xs.append(X_scaled[rd_indicies, :])
        Ys.append(Y[rd_indicies, :])
    Xs.append(X_scaled[sorted_global_mu_vector_indices[-test_size:]])
    Ys.append(Y[sorted_global_mu_vector_indices[-test_size:]])
    test_raw = X[sorted_global_mu_vector_indices[-test_size:]]

    X_train = X_scaled[available_indicies, :]
    Y_train = Y[available_indicies, :]
    return Xs, Ys, X_train, Y_train, test_raw

