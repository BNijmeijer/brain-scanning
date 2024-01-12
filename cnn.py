import os, time
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py as h
import numpy as np
import tensorflow as tf
import keras as ks
from keras import models, layers, losses

def build_model():
    print("this should be a CNN")

    model = models.Sequential()
    model.add(layers.Conv2D(16,(4,25), activation='relu', input_shape=(248,4453,1)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(32,(4,4), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(32,(4,4), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(4))

    model.summary()
    model.compile(optimizer='adam',
                  loss = losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):

    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=10,
                        validation_data=(X_val,y_val))
    
    return history
