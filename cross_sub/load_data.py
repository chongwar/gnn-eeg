import os
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore", message="Numerical issues were encountered ")


def scale_data(data):
    for i in range(data.shape[0]):
        data[i, ...] = preprocessing.scale(data[i, ...])
        # data[i, ...] = preprocessing.minmax_scale(data[i, ...])
        # data[i, ...] = preprocessing.maxabs_scale(data[i, ...])


def load_corss_sub_data(subject_id=1):
    subjects = [i for i in range(6, 14)]
    
    x_train, y_train = None, None
    for subject in subjects:
        data_dir = f'data/s{subject:>02d}'
        
        # shape of x:(N, T, C)
        x = np.load(f'../data/s{subject_id:>02d}/x_2.npy').astype(np.float32)
        y = np.load(f'../data/s{subject_id:>02d}/y_2.npy').astype(np.int64)
        # (N, T, C) --> (N, T/4, C)
        x = x[:, range(0, x.shape[1], 4), :]
        scale_data(x)
        
        if subject != subject_id:
            if x_train is None:
                x_train, y_train = x, y
            else:
                x_train = np.concatenate((x_train, x), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)
        else:
            x_test, y_test = x, y

    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
    print(f'Processed train data shape: {x_train.shape}')
    print(f'Processed test  data shape: {x_test.shape}')
    return (x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    _data_dir = '/home/chongwar/code/eeg/GNN/eeg_test/data'
    _subject_id = [6, 7]

    for subject_id in tqdm(_subject_id):
        # data_dir = os.path.join(_data_dir, f's{subject_id:>02d}', 'original')
        # block_num = len(os.listdir(data_dir)) // 2
        # save_dir = os.path.join(_data_dir, f's{subject_id:>02d}')
        # _load_data(data_dir, save_dir, block_num)

        load_corss_sub_data(subject_id)

    # subject_id = 6
    # x_train, x_test, y_train, y_test = load_data(subject_id)
