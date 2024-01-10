import os
import tensorflow as tf
import warnings

# Suppress TensorFlow logging except for errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = no INFO, 2 = no INFO and WARNING, 3 = no INFO, WARNING, and ERROR
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # For TensorFlow 1.x
# Suppress Python warnings
warnings.filterwarnings('ignore')

from extract_data import extract_data
from evaluate import evaluate_model
import CNN



def main():
    # Get data
    cross_folder = './data/Cross/'
    intra_folder = './data/Intra/'
    # train_data_cross, train_labels_cross = extract_data(os.path.join(cross_folder,'train'))
    train_data_intra, train_labels_infra = extract_data(os.path.join(intra_folder,'train'))
    # test_data_cross, test_labels_cross = extract_data(os.path.join(cross_folder,'test'))
    test_data_intra, test_labels_infra = extract_data(os.path.join(intra_folder,'test'))

    print(train_data_intra.shape)
    print(train_labels_infra)

    model_intra, history_intra = CNN.train_model(train_data_intra, train_labels_infra)

    evaluate_model(model_intra, test_data_intra, test_labels_infra)  # Labels are generated inside the evaluate_model function

if __name__ == '__main__':
    main()