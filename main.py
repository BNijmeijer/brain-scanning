import h5py
import numpy as np
from scipy.stats import zscore
import os
import tensorflow as tf
import matplotlib.pyplot  as plt
import preprocessing as p
import models as m

def main():
    y_intra_train, x_intra_train, y_intra_test, x_intra_test = p.load_intra()

    y_cross_train, x_cross_train = p.load_cross_train()
    y1_cross_test, x1_cross_test = p.load_cross_test("1") # corresponds to the folders
    y2_cross_test, x2_cross_test = p.load_cross_test("2")
    y3_cross_test, x3_cross_test = p.load_cross_test("3")

    amount = 5

    X_train_intra, y_train_intra,\
        X_val_intra, y_val_intra,\
        X_test_intra, y_test_intra,\
        X_train_cross, y_train_cross,\
        X_val_cross, y_val_cross,\
        X1_test_cross, y1_test_cross, \
        X2_test_cross, y2_test_cross, \
        X3_test_cross, y3_test_cross = p.preprocess_all(x_intra_train, y_intra_train, \
                                                        x_intra_test, y_intra_test, \
                                                        x_cross_train, y_cross_train, \
                                                        x1_cross_test, y1_cross_test, \
                                                        x2_cross_test, y2_cross_test, \
                                                        x3_cross_test, y3_cross_test)

    # Splits the files into <amount> files and zscales these
    #x_train = p.preprocess_multiple(x_train, amount)
    #x_test = p.preprocess_multiple(x_test, amount)

    # The repeat is needed to handle the creation of additional files from one file (Het idee van Bas van afgelopen woensdag)
    #y_train = np.repeat(p.relabel_all(y_train), amount)
    #y_test = np.repeat(p.relabel_all(y_test), amount)

    #y_train = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3])
    #y_test = np.array([0,0,1,1,2,2,3,3])
    print(np.shape(X_train_intra))
    print(np.shape(y_train_intra))
    print(np.shape(X_test_intra))
    print(np.shape(y_test_intra))

    return


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

if __name__ == '__main__':
    main()