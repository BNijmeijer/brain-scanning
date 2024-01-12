import os, time
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py as h
import numpy as np
import tensorflow as tf
import keras as ks
from keras import models, layers, losses

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

def train_model(model, x_train, y_train):
    
    return #history

def scale_data(train_data, test_data):
    min_value = np.minimum(np.min(train_data), np.min(test_data))
    max_value = np.maximum(np.max(train_data), np.max(test_data))
    train_data = (train_data - min_value) * max_value
    test_data = (test_data - min_value) * max_value

    assert(np.all(map(lambda x: x>=0 and x<=1, train_data)))
    assert(np.all(map(lambda x: x>=0 and x<=1, test_data)))
    return train_data, test_data

def main():
    
    # Fetch raw data
    cross_folder = './data/Cross/'
    intra_folder = './data/Intra/'
#    train_data_cross = extract_data(os.path.join(cross_folder,'train'))
    train_data_intra, train_labels_intra = extract_data(os.path.join(intra_folder,'train'))
    test_data_intra, test_labels_intra = extract_data(os.path.join(intra_folder,'test'))

    # Rescale data to [0,1] interval
    train_data_intra, test_data_intra = scale_data(train_data_intra, test_data_intra)

    # Convert into TF datasets
    train_set = tf.data.Dataset.from_tensor_slices((train_data_intra, train_labels_intra))
    test_set = tf.data.Dataset.from_tensor_slices((test_data_intra, test_labels_intra))

    model = models.Sequential()
    model.add(layers.Conv1D(16,4, activation='relu', input_shape=(248,35624)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32,4, activation='relu'))
    model.add(layers.MaxPooling1D(4))
    model.add(layers.Conv1D(32,4, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(4))

    model.summary()
#    model_history = train_model(model, train_data_intra, train_labels_intra)
    model.compile(optimizer='adam',
                  loss = losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['accuracy'])
    history = model.fit(train_set, epochs=10,
                        validation_data=test_set)


if __name__ == '__main__':
    main()