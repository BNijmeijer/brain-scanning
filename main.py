import h5py
import numpy as np
from scipy.stats import zscore
import os
import tensorflow as tf
import matplotlib.pyplot  as plt
import preprocessing as p


#y_train, x_train, y_test, x_test = p.load_intra()

y_train, x_train = p.load_cross_train()
y_test, x_test = p.load_cross_test("1")


x_train = p.preprocess(x_train, 5)
x_test = p.preprocess(x_test, 5)

y_train = p.relabel_all(y_train)
y_test = p.relabel_all(y_test)
print(y_test)

#y_train = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3])
#y_test = np.array([0,0,1,1,2,2,3,3])

# Variables
INPUT_SHAPE = (248, 7125, 1)
FILTER1_SIZE = 16
FILTER2_SIZE = 64
FILTER_SHAPE = (3, 3)
POOL_SHAPE = (2, 25)
FULLY_CONNECT_NUM = 64
NUM_CLASSES = 4


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu', input_shape=INPUT_SHAPE))
model.add(tf.keras.layers.MaxPooling2D(POOL_SHAPE))
model.add(tf.keras.layers.Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(POOL_SHAPE))
model.add(tf.keras.layers.Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu'))
# model.add(tf.keras.layers.LSTM(32, return_sequences=True))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(FULLY_CONNECT_NUM, activation='relu'))
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, 
                    validation_data=(x_test, y_test))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
#plt.show()
# test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)



#test_acc

# mat = get_matrix(filename_path)
# mat2 = downsample_matrix(mat, 5)
# mat3 = zscale(mat2)
# print(mat2.shape)
# print(mat.shape)
# print(mat3[:,0])






# model = tf.keras.models.Sequential()

# # Convolutional layers
# model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(248, 7125, 1)))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((4, 4)))
# model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# # Flatten layer
# model.add(tf.keras.layers.Flatten())

# # Dense layers
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# # model.add(tf.keras.layers.Dense(16, activation='relu'))

# # Output layer
# model.add(tf.keras.layers.Dense(4, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Display the model summary
# model.summary()

# history = model.fit(x_train, y_train, epochs=5, 
#                     validation_data=(x_test, y_test))