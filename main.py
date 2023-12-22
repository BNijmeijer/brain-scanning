import os, time
from tqdm import tqdm
import h5py as h
import numpy as np
import tensorflow as tf

def get_dataset_name(path_with_dir : str):
    dataset_name = path_with_dir.split('/')[-1]
    dataset_name = '_'.join(dataset_name.split('_')[:-1])   # Removes final identifier number
    return dataset_name

def extract_data(data_folder : str):
    print(f'Extracting data from {data_folder}...')
    full_data = []
    for file in tqdm(os.listdir(data_folder)):
        full_path = '/'.join([data_folder, file])
        with h.File(full_path,'r') as f:
            dataset_name = get_dataset_name(full_path)
            data_matrix = f.get(dataset_name)[()]
            full_data.append(data_matrix)
    print(np.shape(full_data))
    return full_data


def main():
    print('hi i guess')
    cross_folder = './data/Cross/'
    intra_folder = './data/Intra/'
    train_data_cross = extract_data(os.path.join(cross_folder,'train'))
    train_data_intra = extract_data(os.path.join(intra_folder,'train'))

    

if __name__ == '__main__':
    main()