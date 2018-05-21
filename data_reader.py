# data_reader.py
# Created by abdularis on 24/04/18

import math
import h5py
import numpy as np


def read_data_set(path, batch_size):
    file = h5py.File(path, 'r')
    train = DataSet(file, 'train_images', 'train_labels', batch_size)
    val = DataSet(file, 'val_images', 'val_labels', batch_size)
    test = DataSet(file, 'test_images', 'test_labels', batch_size)

    return train, val, test


class DataSet(object):

    def __init__(self, h5file, dataset_name_images, dataset_name_labels, batch_size):
        self.file = h5file
        self.dataset_name_images = dataset_name_images
        self.dataset_name_labels = dataset_name_labels
        self.batch_size = batch_size
        self.current_batch = 0
        self.data_set_size = len(self.file[self.dataset_name_images])
        self.batch_count = math.ceil(self.data_set_size / self.batch_size)

    def _next_start_end_index(self):
        if self.current_batch >= self.batch_count:
            self.current_batch = 0

        start_idx = min(self.current_batch * self.batch_size, self.data_set_size)
        end_idx = min(start_idx + self.batch_size, self.data_set_size)
        self.current_batch += 1
        return start_idx, end_idx

    def get_one_hot_labels(self):
        one_hot = {}
        for label in self.file['label_encoding']:
            one_hot[label] = self.file['label_encoding'][label].value
        return one_hot

    def get_labels(self):
        one_hot = self.get_one_hot_labels()
        labels = ['' for _ in range(len(one_hot))]
        for label in one_hot:
            max_idx = np.argmax(one_hot[label])
            labels[max_idx] = label
        return labels

    def next_batch(self):
        start_idx, end_idx = self._next_start_end_index()
        return self.file[self.dataset_name_images][start_idx:end_idx], self.file[self.dataset_name_labels][start_idx:end_idx]
