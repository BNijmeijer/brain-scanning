import os, time
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py as h
import numpy as np
import tensorflow as tf
import keras as ks
from keras import models, layers, losses

def main():
    print("this should be a CNN")

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