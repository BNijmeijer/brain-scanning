import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot  as plt
import preprocessing as p
import models as m
from tqdm import tqdm

def main():

    if not os.path.exists("./prepped_data_intra.npz"):
        # Load and preprocess data from scratch
        y_intra_train, x_intra_train, y_intra_test, x_intra_test = p.load_intra()

        y_cross_train, x_cross_train = p.load_cross_train()
        y1_cross_test, x1_cross_test = p.load_cross_test("1") # corresponds to the folders
        y2_cross_test, x2_cross_test = p.load_cross_test("2")
        y3_cross_test, x3_cross_test = p.load_cross_test("3")

        X_train_intra, y_train_intra, X_val_intra, y_val_intra = p.prep(x_intra_train, y_intra_train, build_validation_set = True)
        X_test_intra, y_test_intra = p.prep(x_intra_test, y_intra_test)
        X_train_cross, y_train_cross, X_val_cross, y_val_cross = p.prep(x_cross_train, y_cross_train, build_validation_set = True)
        X_test1_cross, y_test1_cross = p.prep(x1_cross_test, y1_cross_test)
        X_test2_cross, y_test2_cross = p.prep(x2_cross_test, y2_cross_test)
        X_test3_cross, y_test3_cross = p.prep(x3_cross_test, y3_cross_test)

        # Relabel here for speed purposes
        y_train_intra = p.relabel_all(y_train_intra)
        y_val_intra = p.relabel_all(y_val_intra)
        y_test_intra = p.relabel_all(y_test_intra)
        y_train_cross = p.relabel_all(y_train_cross)
        y_val_cross = p.relabel_all(y_val_cross)
        y_test1_cross = p.relabel_all(y_test1_cross)
        y_test2_cross = p.relabel_all(y_test2_cross)
        y_test3_cross = p.relabel_all(y_test3_cross)

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
    intra_v_cross(np.transpose(X_train_intra,axes=(0,2,1)),y_train_intra,
                  np.transpose(X_val_intra,axes=(0,2,1)),y_val_intra,
                  np.transpose(X_test_intra,axes=(0,2,1)),y_test_intra,
                  np.transpose(X_train_cross,axes=(0,2,1)),y_train_cross,
                  np.transpose(X_val_cross,axes=(0,2,1)),y_val_cross,
                  np.transpose(X_test1_cross,axes=(0,2,1)),y_test1_cross,
                  np.transpose(X_test2_cross,axes=(0,2,1)),y_test2_cross,
                  np.transpose(X_test3_cross,axes=(0,2,1)),y_test3_cross)
    return

    intra_model = m.rnnmodel(input_shape=(4453,248))
    intra_model.build()
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    intra_model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    intra_model.summary()

    #epochs is maybe een hyperparameter?
    intra_history = intra_model.fit(np.transpose(X_train_intra,axes=(0,2,1)), y_train_intra, epochs=5, 
                       validation_data=(np.transpose(X_val_intra, axes=(0,2,1)), y_val_intra))
    intra_test_loss, intra_test_acc = intra_model.evaluate(np.transpose(X_test_intra, axes=(0,2,1)), y_test_intra, verbose = 2)

    cross_model = m.rnnmodel(input_shape=(4453,248))
    cross_model.build()
    cross_model.compile(optimizer='adam',
                        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics = ['accuracy'])
    cross_model.summary()
    cross_history = cross_model.fit(np.transpose(X_train_cross, axes=(0,2,1)), y_train_cross, epochs=5,
                              validation_data = (np.transpose(X_val_cross, axes = (0,2,1)), y_val_cross))
    cross_test1_loss, cross_test1_acc = cross_model.evaluate(np.transpose(X_test1_cross, axes=(0,2,1)), y_test1_cross, verbose=2)
    cross_test2_loss, cross_test2_acc = cross_model.evaluate(np.transpose(X_test2_cross, axes=(0,2,1)), y_test2_cross, verbose=2)
    cross_test3_loss, cross_test3_acc = cross_model.evaluate(np.transpose(X_test3_cross, axes=(0,2,1)), y_test3_cross, verbose=2)

    # Plots the two accuracies against the epochs
    plt.plot(intra_history.history['accuracy'], label='Accuracy, intra')
    plt.plot(intra_history.history['val_accuracy'], label = 'Validation accuracy, intra')
    plt.plot(cross_history.history['accuracy'], label='Accuracy, cross')
    plt.plot(cross_history.history['val_accuracy'], label = 'Validation accuracy, cross')
    plt.xlabel('Epoch')
    plt.xticks(range(5))
    plt.ylabel('Accuracy')
    plt.ylim([0.4, 1.05])
    plt.legend(loc='lower right')
    plt.show()

def cv_rnn(X_train, y_train, X_val, y_val, X_test1, y_test1, X_test2, y_test2, X_test3, y_test3):
    test_accs = []
    for i in np.arange(3,7):
        for j in np.arange(3,i+1):
                model = m.rnnmodel(input_shape=(4453,248), units_1=2**i, units_2=2**j)
                model.compile(optimizer='adam',
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                metrics=['accuracy'])
                model.summary()

                model.fit(X_train, y_train, epochs=5, 
                                validation_data=(X_val, y_val), verbose = 2)
                _, intra_test1_acc = model.evaluate(X_test1, y_test1, verbose = 2)
                _, intra_test2_acc = model.evaluate(X_test2, y_test2, verbose = 2)
                _, intra_test3_acc = model.evaluate(X_test3, y_test3, verbose = 2)
                
                test_accs.append(np.array([intra_test1_acc, intra_test2_acc, intra_test3_acc]))
    print(test_accs)
    return

def intra_v_cross(X_train_intra, y_train_intra, X_val_intra, y_val_intra, X_test_intra, y_test_intra, X_train_cross, y_train_cross, X_val_cross, y_val_cross, X_test1_cross, y_test1_cross, X_test2_cross, y_test2_cross, X_test3_cross, y_test3_cross):
    intra_test_accs = []
    cross_test_accs = []
    for _ in tqdm(range(10)):
            intra_model = m.rnnmodel(input_shape=(4453,248))
            intra_model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])
            intra_model.fit(X_train_intra, y_train_intra, epochs=5, 
                            validation_data=(X_val_intra, y_val_intra), verbose = 0)
            _, intra_test_acc = intra_model.evaluate(X_test_intra, y_test_intra, verbose =0)
                
    intra_test_accs.append(intra_test_acc)
    
    for _ in tqdm(range(10)):
        model = m.rnnmodel(input_shape=(4453,248))
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        model.fit(X_train_cross, y_train_cross, epochs=5, 
                        validation_data=(X_val_cross, y_val_cross), verbose = 0)
        _, intra_test1_acc = model.evaluate(X_test1_cross, y_test1_cross, verbose = 0)
        _, intra_test2_acc = model.evaluate(X_test2_cross, y_test2_cross, verbose = 0)
        _, intra_test3_acc = model.evaluate(X_test3_cross, y_test3_cross, verbose = 0)
        
        cross_test_accs.append([intra_test1_acc, intra_test2_acc, intra_test3_acc])
    
    cross_test_accs = np.transpose(np.array(cross_test_accs))
    intra_avg = np.average(intra_test_accs)
    intra_stddev = np.std(intra_test_accs)
    cross_avgs = [np.average(x) for x in cross_test_accs]
    cross_stddevs = [np.std(x) for x in cross_test_accs]
    print(f'Intra average: {intra_avg}, stddev: {intra_stddev}')
    print(f'Cross1 average: {cross_avgs[0]}, stddev: {cross_stddevs[0]}')
    print(f'Cross2 average: {cross_avgs[1]}, stddev: {cross_stddevs[1]}')
    print(f'Cross3 average: {cross_avgs[2]}, stddev: {cross_stddevs[2]}')


if __name__ == '__main__':
    main()