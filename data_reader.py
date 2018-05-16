# data_reader.py
# Created by abdularis on 24/04/18

import math
import h5py
import numpy as np


class DataSet(object):

    def __init__(self, filename, batch_size):
        self.file = h5py.File(filename, mode='r')
        self.batch_size = batch_size
        self.current_batch = 0
        self.train_size = len(self.file['train_images'])
        self.batch_count = math.ceil(self.train_size / self.batch_size)

    def _next_start_end_index(self):
        if self.current_batch >= self.batch_count:
            self.current_batch = 0

        start_idx = min(self.current_batch * self.batch_size, self.train_size)
        end_idx = min(start_idx + self.batch_size, self.train_size)
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
        return self.file['train_images'][start_idx:end_idx], self.file['train_labels'][start_idx:end_idx]

    def next_batch_val(self):
        start_idx, end_idx = self._next_start_end_index()
        return self.file['val_images'][start_idx:end_idx], self.file['val_labels'][start_idx:end_idx]

    def next_batch_test(self):
        start_idx, end_idx = self._next_start_end_index()
        return self.file['test_images'][start_idx:end_idx], self.file['test_labels'][start_idx:end_idx]

    def val_data(self):
        return self.file['val_images'], self.file['val_labels']

    def test_data(self):
        return self.file['test_images'], self.file['test_labels']
