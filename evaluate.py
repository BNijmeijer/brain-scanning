import numpy as np
import logging
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def evaluate_model(model, test_data, test_labels):

    logging.info("Starting model evaluation...")

    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)

    cm = confusion_matrix(test_labels, predicted_classes)
    logging.info(f"Confusion Matrix:\n{cm}")

    cr = classification_report(test_labels, predicted_classes)
    logging.info(f"Classification Report:\n{cr}")

    loss, accuracy = model.evaluate(test_data, test_labels)
    logging.info(f"Test Loss: {loss}")
    logging.info(f"Test Accuracy: {accuracy}")

    logging.info("Model evaluation completed.")
