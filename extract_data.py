import os
from tqdm import tqdm
import h5py as h
import numpy as np

def get_task_type(file_name: str):
    parts = file_name.split('_')[:-2]
    task_type = '_'.join(parts)
    return task_type

def get_dataset_name(file_name: str):
    parts = file_name.split('_')
    dataset_name = '_'.join(parts[:-1])
    return dataset_name

def extract_data(data_folder: str):
    full_data = []
    labels = []
    task_labels = {'rest': 0, 'task_motor': 1, 'task_story_math': 2, 'task_working_memory': 3}

    for file in tqdm(os.listdir(data_folder)):
        full_path = os.path.join(data_folder, file)
        print(full_path)
        with h.File(full_path, 'r') as f:
            task_type = get_task_type(file)
            data_matrix = f.get(get_dataset_name(file))
            if data_matrix is not None:
                full_data.append(data_matrix[()])
                labels.append(task_labels.get(task_type, -1))

    return np.expand_dims(np.stack(full_data)[:, :, ::10], axis=-1), np.array(labels)
