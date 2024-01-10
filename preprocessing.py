import os
import h5py
import numpy as np
import re

#
# Extract the label from the dataset using the file name
#
# 0 = rest, 1 = motor, 2 = story_math, 3 = working_memory
#

def get_label(filename):
    rest_pattern = re.compile(r'^rest_\d+(_\d+)?\.h5$')
    task_motor_pattern = re.compile(r'^task_motor_\d+(_\d+)?\.h5$')
    task_story_math_pattern = re.compile(r'^task_story_math_\d+(_\d+)?\.h5$')
    task_working_memory_pattern = re.compile(r'^task_working_memory_\d+(_\d+)?\.h5$')

    if rest_pattern.match(filename):
        return 0
    elif task_motor_pattern.match(filename):
        return 1
    elif task_story_math_pattern.match(filename):
        return 2
    elif task_working_memory_pattern.match(filename):
        return 3

def get_dataset_name(filename_with_dir):
    filename_without_dir = filename_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

#
# Downsamples the matrices with a gives factor
#

def downsample_matrix(matrix, factor):
    downsampled_matrix = matrix[:, ::factor]
    return downsampled_matrix

#
# Scale the values to a number between 0 and 1
#

def min_max_scaling(matrix):
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)
    scaled_matrix = (matrix - min_vals) / (max_vals - min_vals)
    return scaled_matrix

#
# Create a matrix of all datapoints combined. Each row is a datapoint. The first column in every row is the respective label.
#

def get_matrix(directory_path, downsample_factor):
    matrices = []
    list_of_labels = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        dataset_name = get_dataset_name(file_path)
    
        with h5py.File(file_path, 'r') as f:
            matrix = f.get(dataset_name)[:]
            matrix = downsample_matrix(matrix, downsample_factor)
            matrix = min_max_scaling(matrix)
            list_of_labels.extend([get_label(filename)] * matrix.shape[1])  # Create a list of labels which
            matrices.append(matrix)

    labels_array = np.array(list_of_labels)
    labels_array = labels_array.reshape((1, labels_array.shape[0]))  # Reshape labels to be a row vector

    combined_matrix = np.vstack([labels_array, np.concatenate(matrices, axis=1)])  # Combine matrices and labels into a single matrix

    # Separate labels and data
    labels = combined_matrix[0]
    data = combined_matrix[1:]

    # Apply min-max scaling only to the data
    scaled_data = min_max_scaling(data)

    # Combine the scaled data and labels back into a single matrix
    scaled_matrix = np.vstack([labels, data])

    flipped_matrix = scaled_matrix.T  # Transpose the scaled matrix to swap rows and columns

    print("Created flipped matrix for :", directory_path, "with shape:", flipped_matrix.shape)

    return flipped_matrix