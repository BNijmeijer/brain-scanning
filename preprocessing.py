import h5py
import numpy as np
from scipy.stats import zscore
import os

def get_dataset_name(filename_with_dir):
    filename_without_dir = filename_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

def get_matrix(filename):
    with h5py.File(filename, "r") as f:
        dataset_name = get_dataset_name(filename)
        matrix = f.get(dataset_name)[()]
        return matrix

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

def zs_scale(mat):
    return zscore(mat)

def downsample(X, y, downsample_factor=8):
    # creates a larger, downsampled dataset.
    # NOTE: Ensure that downsample_factor is a proper divisor of the number of time steps.
    X_train = []
    y_train = []

    for (i,data) in enumerate(X):
        X_train.extend([data[:,j::downsample_factor] for j in range(downsample_factor)])
        y_train.extend(np.repeat(y[i], downsample_factor))

    return X_train, y_train

def build_val(X, y, val_split = 8):
    # Take every 8th example, and turn into validation dataset
    X_t = []
    y_t = []
    X_val = []
    y_val = []
    for i in range(np.shape(X)[0]):
        if i % val_split == 0:
            X_val.append(X[i])
            y_val.append(y[i])
        else:
            X_t.append(X[i])
            y_t.append(y[i])
    return np.array(X_t), np.array(X_val), np.array(y_t), np.array(y_val)

def prep(X, y, build_validation_set = False):
    # Create downsampled and scaled dataset. Also build validation dataset if necessary.
    X_ds, y_ds = downsample(X, y)
    X_scale = zscore(X_ds)
    if build_validation_set:
        X_train, X_val, y_train, y_val = build_val(X_scale, y_ds)
        return X_train, y_train, X_val, y_val
    else:
        return X_scale, y_ds
