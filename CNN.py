import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging

# Make sure logging also states the time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(data, downsampling_factor=10):
    logging.info("Starting data preprocessing...")
    data = np.array(data)
    


    # Add a channel dimension because the CNN expects a channel dimension
    # processed_data = np.expand_dims(processed_data, axis=-1)

    logging.info(f"Data preprocessing completed. Shape after preprocessing: {data.shape}")
    return data

def create_cnn_model(input_shape):
    logging.info(f"Creating CNN model with input shape: {input_shape}")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation='softmax') # 4 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logging.info("CNN model created.")
    return model

def train_model(train_data, train_labels, epochs=10, batch_size=32):
    logging.info("Starting model training process...")
    
    processed_train_data = preprocess_data(train_data)

    # Create and train the CNN model
    model = create_cnn_model(processed_train_data.shape[1:])
    history = model.fit(processed_train_data, train_labels, epochs=epochs, batch_size=batch_size)

    logging.info("Model training completed.")
    return model, history
