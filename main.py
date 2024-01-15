import h5py
import numpy as np
from scipy.stats import zscore
import os
import tensorflow as tf
import matplotlib.pyplot  as plt
import preprocessing as p
import models as m

#y_train, x_train, y_test, x_test = p.load_intra()

y_train, x_train = p.load_cross_train()
y_test, x_test = p.load_cross_test("1") # corresponds to the folders

amount = 5

# Splits the files into <amount> files and zscales these
x_train = p.preprocess_multiple(x_train, amount)
x_test = p.preprocess_multiple(x_test, amount)

# The repeat is needed to handle the creation of additional files from one file (Het idee van Bas van afgelopen woensdag)
y_train = np.repeat(p.relabel_all(y_train), amount)
y_test = np.repeat(p.relabel_all(y_test), amount)

#y_train = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3])
#y_test = np.array([0,0,1,1,2,2,3,3])



model = m.rnnmodel()
model.build()
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
model.summary()

#epochs is maybe een hyperparameter?
history = model.fit(x_train, y_train, epochs=5, 
                    validation_data=(x_test, y_test))


# Plots the two accuracies against the epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()




