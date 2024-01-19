import tensorflow as tf


# Variables
INPUT_SHAPE = (248, 7124, 1)
FILTER1_SIZE = 16
FILTER2_SIZE = 64
FILTER_SHAPE = (3, 3)
POOL_SHAPE = (2, 25)
FULLY_CONNECT_NUM = 64
NUM_CLASSES = 4

def cnnmodel(input_shape=INPUT_SHAPE):
    inputs = input_shape
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu', input_shape=inputs))
    model.add(tf.keras.layers.MaxPooling2D(POOL_SHAPE))
    model.add(tf.keras.layers.Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(POOL_SHAPE))
    model.add(tf.keras.layers.Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(FULLY_CONNECT_NUM, activation='relu'))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    return model

RNN_UNITS = 32

def rnnmodel(input_shape = INPUT_SHAPE, units_1 = RNN_UNITS, units_2 = RNN_UNITS):
    INPUT_SHAPE = input_shape
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(units=units_1, activation='relu',input_shape=INPUT_SHAPE, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(units=units_2, activation='relu'))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    
    return model