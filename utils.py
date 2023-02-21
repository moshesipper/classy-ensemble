# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# utilities

import argparse
import numpy as np
from random import choices
from string import ascii_lowercase
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pmlb import fetch_data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', dest='dataset', type=str, action='store', default='auto',
                        help='Dataset to use (default: auto)')
    parser.add_argument('-resdir', dest='resdir', type=str, action='store', default='Results',
                        help='directory where results are placed (default: Results)')
    parser.add_argument('-reps', dest='n_replicates', type=int, action='store', default=30,
                        help='number of replicate runs (default: 30)')

    args = parser.parse_args()
    if None in [getattr(args, arg) for arg in vars(args)]:
        parser.print_help()
        exit()

    dataset, resdir, n_replicates = args.dataset, args.resdir, args.n_replicates
    return dataset, resdir, n_replicates


def rndstr(n=6):
    return ''.join(choices(ascii_lowercase, k=n))


def load_dataset(dataset):
    X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='../datasets/pmlb')
    y = LabelEncoder().fit_transform(y)  # To avoid XGBoost error: Invalid classes inferred from unique values of `y`. Expected: [0 1], got [1 2]

    n_samples, n_features, n_classes = X.shape[0], X.shape[1], len(np.unique(y))
    return X, y, n_samples, n_features, n_classes


def train_val_test_split(X, y, val_size=0.2, test_size=0.2):
    # default is train/val/test: 60%-20%-20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=1)

    # scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # scale training data
    X_val = scaler.transform(X_val)  # use SAME scaler as one fitted to training data
    X_test = scaler.transform(X_test)  # use SAME scaler as one fitted to training data

    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_val.shape[0] + y_test.shape[0] == y.shape[0]

    return X_train, X_val, X_test, y_train, y_val, y_test


def acc_per_class(y_true, y_pred, n_classes):
    labels = range(n_classes)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    res = matrix.diagonal() / matrix.sum(axis=1)
    res[np.isnan(res)] = 0
    return list(res)

