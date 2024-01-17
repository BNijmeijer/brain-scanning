import h5py
import numpy as np
from scipy.stats import zscore
import os
import tensorflow as tf
import matplotlib.pyplot  as plt
import preprocessing as p
import models as m

def main():
    if not os.path.exists("./prepped_data_intra.npz"):
        # Load and preprocess data from scratch
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
            X_test1_cross, y_test1_cross, \
            X_test2_cross, y_test2_cross, \
            X_test3_cross, y_test3_cross = p.preprocess_all(x_intra_train, y_intra_train, \
                                                            x_intra_test, y_intra_test, \
                                                            x_cross_train, y_cross_train, \
                                                            x1_cross_test, y1_cross_test, \
                                                            x2_cross_test, y2_cross_test, \
                                                            x3_cross_test, y3_cross_test)
        # Save data for later use
        np.savez('prepped_data_intra',
                 X_train_intra = X_train_intra,
                 y_train_intra = y_train_intra,
                 X_val_intra = X_val_intra,
                 y_val_intra = y_val_intra,
                 X_test_intra = X_test_intra,
                 y_test_intra = y_test_intra)
        np.savez('prepped_data_cross',
                 X_train_cross = X_train_cross,
                 y_train_cross = y_train_cross,
                 X_val_cross = X_val_cross,
                 y_val_cross = y_val_cross,
                 X_test1_cross = X_test1_cross,
                 y_test1_cross = y_test1_cross,
                 X_test2_cross = X_test2_cross,
                 y_test2_cross = y_test2_cross,
                 X_test3_cross = X_test3_cross,
                 y_test3_cross = y_test3_cross
                 )
    else: # Use existing data
        with np.load('./prepped_data_intra.npz') as intra_file:
            X_train_intra = intra_file['X_train_intra']
            y_train_intra = intra_file['y_train_intra']
            X_val_intra = intra_file['X_val_intra']
            y_val_intra = intra_file['y_val_intra']
            X_test_intra = intra_file['X_test_intra']
            y_test_intra = intra_file['y_test_intra']
        with np.load('./prepped_data_cross.npz') as cross_file:
            X_train_cross = cross_file['X_train_cross']
            y_train_cross = cross_file['y_train_cross']
            X_val_cross = cross_file['X_val_cross']
            y_val_cross = cross_file['y_val_cross']
            X_test1_cross = cross_file['X_test1_cross']
            y_test1_cross = cross_file['y_test1_cross']
            X_test2_cross = cross_file['X_test2_cross']
            y_test2_cross = cross_file['y_test2_cross']
            X_test3_cross = cross_file['X_test3_cross']
            y_test3_cross = cross_file['y_test3_cross']
        
    # Splits the files into <amount> files and zscales these
    #x_train = p.preprocess_multiple(x_train, amount)
    #x_test = p.preprocess_multiple(x_test, amount)

    # The repeat is needed to handle the creation of additional files from one file (Het idee van Bas van afgelopen woensdag)
    #y_train = np.repeat(p.relabel_all(y_train), amount)
    #y_test = np.repeat(p.relabel_all(y_test), amount)

    y_train_intra = p.relabel_all(y_train_intra)
    y_val_intra = p.relabel_all(y_val_intra)
    y_test_intra = p.relabel_all(y_test_intra)
    y_train_cross = p.relabel_all(y_train_cross)
    y_val_cross = p.relabel_all(y_val_cross)
    y_test1_cross = p.relabel_all(y_test1_cross)
    y_test2_cross = p.relabel_all(y_test2_cross)
    y_test3_cross = p.relabel_all(y_test3_cross)

    #y_train = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3])
    #y_test = np.array([0,0,1,1,2,2,3,3])
    print(np.shape(X_train_intra))
    print(np.shape(y_train_intra))
    print(np.shape(X_val_intra))
    print(np.shape(y_val_intra))
    print(np.shape(X_test_intra))
    print(np.shape(y_test_intra))
    print(np.shape(X_train_cross))
    print(np.shape(y_train_cross))
    print(np.shape(X_val_cross))
    print(np.shape(y_val_cross))
    print(np.shape(X_test1_cross))
    print(np.shape(y_test1_cross))
    print(np.shape(X_test2_cross))
    print(np.shape(y_test2_cross))
    print(np.shape(X_test3_cross))
    print(np.shape(y_test3_cross))

#    return


    model = m.rnnmodel(input_shape=(4453,248))
    model.build()
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    model.summary()

    #epochs is maybe een hyperparameter?
    history = model.fit(np.transpose(X_train_intra,axes=(0,2,1)), y_train_intra, epochs=5, 
                        validation_data=(np.transpose(X_val_intra, axes=(0,2,1)), y_val_intra))
    
    test_loss, test_acc = model.evaluate(np.transpose(X_test_intra, axes=(0,2,1)), y_test_intra, verbose = 2)


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