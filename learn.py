from preprocessing import *
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import shuffle

downsample_factor = 4

train_matrix = get_matrix("data/Intra/train/", downsample_factor)
test_matrix =  get_matrix("data/Intra/test/", downsample_factor)

y_train = np.array([row[0] for row in train_matrix])
X_train = np.array([row[1:] for row in train_matrix])
7
y_test = np.array([row[0] for row in test_matrix])
X_test = np.array([row[1:] for row in test_matrix])

#Shuffle the data
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=X_train[0].shape),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=1, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test) 
print(f'Test accuracy: {test_acc}')
