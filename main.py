import os, time
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py as h
import numpy as np
import tensorflow as tf
import keras as ks
from keras import models, layers, losses

import cnn, rnn

def get_dataset_name(path_with_dir : str):
    dataset_name = path_with_dir.split('/')[-1]
    dataset_name = '_'.join(dataset_name.split('_')[:-1])   # Removes final identifier number
    return dataset_name

def extract_data(data_folder : str):
    print(f'Extracting data from {data_folder}...')
    full_data, labels = [], []
    tr={'rest':0, 'task_motor':1, 'task_story_math':2, 'task_working_memory':3}
    for file in tqdm(os.listdir(data_folder)):
        full_path = '/'.join([data_folder, file])
        label = '_'.join(file.split('_')[:-2])
        with h.File(full_path,'r') as f:
            dataset_name = get_dataset_name(full_path)
            data_matrix = f.get(dataset_name)[()]
            full_data.append(data_matrix)
        labels.append(tr[label])
    print(np.shape(full_data))
    return full_data, labels

def downsample(data, labels, factor):

    sampled_data = []
    sampled_labels = []

    for (i,ex) in enumerate(data):
        label = labels[i]
        downsampled_data = [ex[:,j::factor] for j in range(factor)]
        for d in downsampled_data:
            sampled_data.append(d)
            sampled_labels.append(label)
    
    print(np.shape(sampled_data))
    return sampled_data, sampled_labels

def minmax_scale_data(train_data, test_data):
    """Scales the given datasets to the interval [0,1], inclusive."""
    min_value = np.minimum(np.min(train_data), np.min(test_data))
    max_value = np.maximum(np.max(train_data), np.max(test_data))
    train_data = (train_data - min_value) / max_value
    test_data = (test_data - min_value) / max_value

    assert(np.all(map(lambda x: x>=0 and x<=1, train_data)))
    assert(np.all(map(lambda x: x>=0 and x<=1, test_data)))
    return train_data, test_data

def prep_data(X_train, y_train, X_test, y_test, downsample_factor = 8):

    # Downsample and scale train and test data
    ds_X_train, ds_y_train = downsample(X_train, y_train, downsample_factor)
    ds_X_test, ds_y_test = downsample(X_test, y_test, downsample_factor)
    scaled_X_train, scaled_X_test = minmax_scale_data(ds_X_train, ds_X_test)
    
    # Construct validation set from training set
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for (i, x) in enumerate(scaled_X_train):
        if i % 8 == 0:
            X_val.append(x)
            y_val.append(ds_y_train[i])
        else:
            X_train.append(x)
            y_train.append(ds_y_train[i])
    
    return X_train, y_train, X_val, y_val, scaled_X_test, ds_y_test

def main():
    
    # Fetch raw data
    cross_folder = './data/Cross/'
    intra_folder = './data/Intra/'
#    train_data_cross, train_labels_cross = extract_data(os.path.join(cross_folder,'train'))
    X_train_intra, y_train_intra = extract_data(os.path.join(intra_folder,'train'))
    X_test_intra, y_test_intra = extract_data(os.path.join(intra_folder,'test'))
    X_train_intra, y_train_intra, X_val_intra, y_val_intra, X_test_intra, y_test_intra = prep_data(X_train_intra, y_train_intra, X_test_intra, y_test_intra)

    conv_model = cnn.build_model()
    conv_fit = cnn.train_model(conv_model, [X_train_intra], y_train_intra, [X_val_intra], y_val_intra)

    return

if __name__ == '__main__':
    main()