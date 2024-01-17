import h5py
import numpy as np
from scipy.stats import zscore
import os
import tensorflow as tf
import matplotlib.pyplot  as plt



def get_dataset_name(filename_with_dir):
    filename_without_dir = filename_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

#filename_path = "Intra/train/rest_105923_1.h5"
# with h5py.File(filename_path, "r") as f:
#     dataset_name = get_dataset_name(filename_path)
#     matrix = f.get(dataset_name)[()]
#     print(type(matrix))
#     print(matrix.shape)

def get_matrix(filename):
    with h5py.File(filename, "r") as f:
        dataset_name = get_dataset_name(filename)
        matrix = f.get(dataset_name)[()]
        return matrix

def downsample_matrix(matrix, factor):
    result_matrix = matrix[:,np.arange(0, matrix.shape[1], factor)]
    return result_matrix

def load_intra():
    filenames = os.listdir("./data/Intra/train")
    x_train = [get_matrix("./data/Intra/train/" + name) for name in filenames]
    y_train = ['_'.join(name.split('_')[:-2]) for name in filenames]

    filenames = os.listdir("./data/Intra/test")
    x_test = [get_matrix("./data/Intra/test/" + name) for name in filenames]
    y_test = ['_'.join(name.split('_')[:-2]) for name in filenames]
    return y_train, x_train, y_test, x_test

def load_cross_train():
    filenames = os.listdir("./data/Cross/train")
    x_train = [get_matrix("./data/Cross/train/" + name) for name in filenames]
    y_train = ['_'.join(name.split('_')[:-2]) for name in filenames]
    return y_train, x_train

def load_cross_test(nr):
    filenames = os.listdir("./data/Cross/test"+nr)
    x_train = [get_matrix("./data/Cross/test"+nr+"/" + name) for name in filenames]
    y_train = ['_'.join(name.split('_')[:-2]) for name in filenames]
    return y_train, x_train

def preprocess(matrices, factor):
    mats = [downsample_matrix(matrix, factor) for matrix in matrices]
    zs_mat = [zscore(matrix) for matrix in mats]
    return np.array(zs_mat)

def relabel(s):
    if s.startswith("rest"):
        return 0
    elif s.startswith("task_motor"):
        return 1
    elif s.startswith("task_story"):
        return 2
    elif s.startswith("task_working"):
        return 3
    
def relabel_all(labels):
    lab  = [relabel(s) for s in labels]
    return np.array(lab)

def create_multiple_datasets(matrix, amount):
    res = []

    for i in range(amount):
        res.append(matrix[:,np.arange(i, matrix.shape[1], amount)])

    max_size = min(r.shape[1] for r in res)
    res = [r[:,:max_size] for r in res]

    return res
    
def preprocess_multiple(matrices, factor):
    mats = []
    for matrix in matrices:
        mats.extend(create_multiple_datasets(matrix, factor))
        
    zs_mat = [np.transpose(zscore(matrix)) for matrix in mats]
    return np.array(zs_mat)

def zs_scale(mat):
    return zscore(mat)


def downsample(X, y, downsample_factor=8):
    # creates a larger, downsampled dataset.
    X_train = []
    y_train = []

    for (i,data) in enumerate(X):
        X_train.extend([data[:,j::downsample_factor] for j in range(downsample_factor)])
        y_train.extend(np.repeat(y[i],8))

    return X_train, y_train

def build_val(X,y):
    return X[:-1], X[-1], y[:-1], y[-1]

def preprocess_all(train_intra, labels_train_intra, test_intra, labels_test_intra, train_cross, labels_train_cross, test1_cross, labels_test1_cross, test2_cross, labels_test2_cross, test3_cross, labels_test3_cross):
    # Downsample all data
    X_train_intra, y_train_intra = downsample(train_intra, labels_train_intra)
    X_train_cross, y_train_cross = downsample(train_cross, labels_train_cross)
    X_test_intra, y_test_intra = downsample(test_intra, labels_test_intra)
    X_test1_cross, y1_test_cross = downsample(test1_cross, labels_test1_cross)
    X_test2_cross, y2_test_cross = downsample(test2_cross, labels_test2_cross)
    X_test3_cross, y3_test_cross = downsample(test3_cross, labels_test3_cross)

    # Matrix black magic in order to make z-score scaling work
    # Order: intra (training, test), cross (training, test1, test2, test3)
    full_matrix = []
    delims = []
    full_matrix.extend(X_train_intra)
    delims.append(np.shape(X_train_intra)[0])
    full_matrix.extend(X_test_intra)
    delims.append(delims[-1] + np.shape(X_test_intra)[0])
    full_matrix.extend(X_train_cross)
    delims.append(delims[-1] + np.shape(X_train_cross)[0])
    full_matrix.extend(X_test1_cross)
    delims.append(delims[-1] + np.shape(X_test1_cross)[0])
    full_matrix.extend(X_test2_cross)
    delims.append(delims[-1] + np.shape(X_test2_cross)[0])
    full_matrix.extend(X_test3_cross)

    print(np.shape(full_matrix))

    # Perform z-score scaling
    scaled_data = zs_scale(full_matrix)
    
    # Un-black-magicification
    split_data = np.split(scaled_data, delims, axis=0)
    assert (split_data[0] == X_train_intra)

    # Create validation datasets
    X_train_intra, X_val_intra, y_train_intra, y_val_intra = build_val(scaled_data[0], y_train_intra)
    X_train_cross, X_val_cross, y_train_cross, y_val_cross = build_val(scaled_data[1], y_train_cross)

    return X_train_intra, y_train_intra, X_val_intra, y_val_intra, scaled_data[2], y_test_intra, X_train_cross, y_train_cross, X_val_cross, y_val_cross, scaled_data[3], y1_test_cross, scaled_data[4], y2_test_cross, scaled_data[5], y3_test_cross
    