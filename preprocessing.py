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
    filenames = os.listdir("Intra/train")
    x_train = [get_matrix("Intra/train/" + name) for name in filenames]
    y_train = ['_'.join(name.split('_')[:-2]) for name in filenames]

    filenames = os.listdir("Intra/test")
    x_test = [get_matrix("Intra/test/" + name) for name in filenames]
    y_test = ['_'.join(name.split('_')[:-2]) for name in filenames]
    return y_train, x_train, y_test, x_test

def load_cross_train():
    filenames = os.listdir("Cross/train")
    x_train = [get_matrix("Cross/train/" + name) for name in filenames]
    y_train = ['_'.join(name.split('_')[:-2]) for name in filenames]
    return y_train, x_train

def load_cross_test(nr):
    filenames = os.listdir("Cross/test"+nr)
    x_train = [get_matrix("Cross/test"+nr+"/" + name) for name in filenames]
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
    

