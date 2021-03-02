import sys
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import warnings

warnings.filterwarnings("ignore", message="Numerical issues were encountered ")


def scale_data(data):
    for i in range(data.shape[0]):
        data[i, ...] = preprocessing.scale(data[i, ...])


def load_0529_data(sorted_=True):
    if sorted_:
        x = np.load('data/order/2020_05_29/x_sorted.npy').astype(np.float32)
        y = np.load('data/order/2020_05_29/y_sorted.npy').astype(np.int64)
    else:
        x = np.load('data/order/2020_05_29/x_unsorted.npy').astype(np.float32)
        y = np.load('data/order/2020_05_29/y_unsorted.npy').astype(np.int64)

    print(f'Original EEG data shape: {x.shape}')
    x = np.transpose(x, (0, 2, 1))
    y = y.reshape(-1)
    index = np.arange(0, 1024, 4)
    x = x[:, index, :]

    sss = sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=2)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    scale_data(x_train)
    scale_data(x_test)

    print(f'Processed EEG train data shape: {x_train.shape}')
    print(f'Processed EEG test data shape: {x_test.shape}')

    return (x_train, x_test, y_train, y_test)


def load_concat_eeg_data(date, sorted_=True):
    if sorted_:
        x_1 = np.load(f'data/order/2020_{date}_sorted_01/x_sorted.npy').astype(np.float32)
        y_1 = np.load(f'data/order/2020_{date}_sorted_01/y_sorted.npy').astype(np.int64)
        x_2 = np.load(f'data/order/2020_{date}_sorted_02/x_sorted.npy').astype(np.float32)
        y_2 = np.load(f'data/order/2020_{date}_sorted_02/y_sorted.npy').astype(np.int64)
    else:
        x_1 = np.load(f'data/order/2020_{date}_unsorted_01/x_unsorted.npy').astype(np.float32)
        y_1 = np.load(f'data/order/2020_{date}_unsorted_01/y_unsorted.npy').astype(np.int64)
        x_2 = np.load(f'data/order/2020_{date}_unsorted_02/x_unsorted.npy').astype(np.float32)
        y_2 = np.load(f'data/order/2020_{date}_unsorted_02/y_unsorted.npy').astype(np.int64)

    x = np.concatenate((x_1, x_2))
    y = np.concatenate((y_1, y_2))
    y = y.reshape(-1)
    print(f'Original EEG data shape: {x.shape}')

    # (N, C, T) --> (N, T/4, C)
    x = np.transpose(x, (0, 2, 1))
    x = x[:, range(0, x.shape[1], 8), :]

    # train-test shuffle split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=2)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # preprocess
    scale_data(x_train)
    scale_data(x_test)
    print(f'Processed train data shape: {x_train.shape}')
    print(f'Processed test  data shape: {x_test.shape}')

    return (x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    date = '05_29'
    sorted_ = False
    # sorted_ = False
    x_train, x_test, y_train, y_test = load_0529_data(sorted_)
    # x_train, x_test, y_train, y_test = load_concat_eeg_data(date, sorted_)
    # np.save('x.npy', x_test)
    # np.save('y.npy', y_test)
