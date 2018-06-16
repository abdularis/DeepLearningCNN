# data_reader.py
# Created by abdularis on 24/04/18

import os
import math
import random
import scipy.misc
import numpy as np
import glob


def shuffle(list1, list2):
    temp_list = list(zip(list1, list2))
    random.shuffle(temp_list)
    return zip(*temp_list)


def read_data_set_dir(base_data_split_dir, one_hot_folders_dict, batch_size):
    train_dir = os.path.join(base_data_split_dir, 'train')
    val_dir = os.path.join(base_data_split_dir, 'validation')
    test_dir = os.path.join(base_data_split_dir, 'test')

    train = DirDataSet(batch_size, train_dir, one_hot_folders_dict)
    val = DirDataSet(batch_size, val_dir, one_hot_folders_dict)
    test = DirDataSet(batch_size, test_dir, one_hot_folders_dict)

    return train, val, test


class BaseDataSet(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.current_batch = 0
        self.batch_count = 0
        self._data_set_size = 0

    @property
    def data_set_size(self):
        return self._data_set_size

    @data_set_size.setter
    def data_set_size(self, value):
        self._data_set_size = value
        self.batch_count = math.ceil(self._data_set_size / self.batch_size)

    def _next_start_end_index(self):
        if self.current_batch >= self.batch_count:
            self.current_batch = 0

        start_idx = min(self.current_batch * self.batch_size, self.data_set_size)
        end_idx = min(start_idx + self.batch_size, self.data_set_size)
        self.current_batch += 1
        return start_idx, end_idx

    def next_batch(self):
        start_idx, end_idx = self._next_start_end_index()
        return self._on_next_batch(start_idx, end_idx)

    def _on_next_batch(self, start_idx, end_idx):
        pass


class DirDataSet(BaseDataSet):

    def __init__(self, batch_size, base_dir, one_hot_with_folders):
        super().__init__(batch_size)
        self.file_paths = []
        self.file_labels = []

        for folder, one_hot in one_hot_with_folders.items():
            folder_path = os.path.join(base_dir, folder, '*.jpg')
            img_paths = glob.glob(folder_path)
            self.file_paths.extend(img_paths)
            self.file_labels.extend([one_hot for _ in range(len(img_paths))])

        if len(self.file_paths) != len(self.file_labels):
            print("Error paths and labels don't match!")
            return

        self.file_paths, self.file_labels = shuffle(self.file_paths, self.file_labels)
        self.data_set_size = len(self.file_paths)

    def _read_images(self, start_idx, end_idx):
        imgs = []
        for path in self.file_paths[start_idx:end_idx]:
            imgs.append(scipy.misc.imread(path, mode='RGB'))
        return np.array(imgs)

    def _on_next_batch(self, start_idx, end_idx):
        return self._read_images(start_idx, end_idx), np.array(self.file_labels[start_idx:end_idx])
