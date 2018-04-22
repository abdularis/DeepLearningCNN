# packaging.py
# Created by abdularis on 30/03/18

import os
import random
import scipy.misc
import h5py
import numpy as np
from tqdm import tqdm


LABELS = [
    'backpack',
    'cat',
    'jeep',
    'microwave',
    'mug',
    'teapot'
]

ONE_HOT_IMAGE_LABELS = {
    'backpack':  np.array([1., 0., 0., 0., 0., 0.]),
    'cat':       np.array([0., 1., 0., 0., 0., 0.]),
    'jeep':      np.array([0., 0., 1., 0., 0., 0.]),
    'microwave': np.array([0., 0., 0., 1., 0., 0.]),
    'mug':       np.array([0., 0., 0., 0., 1., 0.]),
    'teapot':    np.array([0., 0., 0., 0., 0., 1.])
}

base_dir = 'dataset/160/'


def shuffle(list1, list2):
    temp_list = list(zip(list1, list2))
    random.shuffle(temp_list)
    return zip(*temp_list)


def get_dataset_paths(train_percent, validation_percent, test_percent):
    if (train_percent + validation_percent + test_percent) != 100:
        print('Train % + validation % + test % != 100%')
        return

    train = {}
    validation = {}
    test = {}

    for category_name in ONE_HOT_IMAGE_LABELS:
        dir_path = os.path.join(base_dir, category_name)

        temp_paths = []
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            temp_paths.append(file_path)

        random.shuffle(temp_paths)

        train_offset = round(len(temp_paths) / 100 * train_percent)
        validation_offset = train_offset + round(len(temp_paths) / 100 * validation_percent)
        test_offset = validation_offset + round(len(temp_paths) / 100 * test_percent)

        train[category_name] = temp_paths[:train_offset]
        validation[category_name] = temp_paths[train_offset:validation_offset]
        test[category_name] = temp_paths[validation_offset:test_offset]

    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    test_paths = []
    test_labels = []

    for category_name in ONE_HOT_IMAGE_LABELS:
        for path in train[category_name]:
            train_paths.append(path)
            train_labels.append(ONE_HOT_IMAGE_LABELS[category_name])
        for path in validation[category_name]:
            val_paths.append(path)
            val_labels.append(ONE_HOT_IMAGE_LABELS[category_name])
        for path in test[category_name]:
            test_paths.append(path)
            test_labels.append(ONE_HOT_IMAGE_LABELS[category_name])

    train_paths, train_labels = shuffle(train_paths, train_labels)
    val_paths, val_labels = shuffle(val_paths, val_labels)
    test_paths, test_labels = shuffle(test_paths, test_labels)

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def package():
    print('[*] Packaging...')

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = get_dataset_paths(80, 10, 10)

    train_shape = (len(train_paths), 160, 160, 3)
    val_shape = (len(val_paths), 160, 160, 3)
    test_shape = (len(test_paths), 160, 160, 3)

    h5_file = h5py.File('dataset.h5', mode='w')
    h5_file.create_dataset('train_images', train_shape, np.uint8)
    h5_file.create_dataset('train_labels', data=train_labels)
    h5_file.create_dataset('val_images', val_shape, np.uint8)
    h5_file.create_dataset('val_labels', data=val_labels)
    h5_file.create_dataset('test_images', test_shape, np.uint8)
    h5_file.create_dataset('test_labels', data=test_labels)

    print('[*] Packaging training images')
    for i in tqdm(range(len(train_paths))):
        h5_file['train_images'][i, ...] = scipy.misc.imread(train_paths[i], mode='RGB')

    print('[*] Packaging validation images')
    for i in tqdm(range(len(val_paths))):
        h5_file['val_images'][i, ...] = scipy.misc.imread(val_paths[i], mode='RGB')

    print('[*] Packaging testing images')
    for i in tqdm(range(len(test_paths))):
        h5_file['test_images'][i, ...] = scipy.misc.imread(test_paths[i], mode='RGB')

    h5_file.close()


def check_package():
    h5_file = h5py.File('dataset.h5', mode='r')

    rand_idx = np.random.random_integers(0, len(h5_file['train_images']), 24)
    images = []
    labels = []
    [images.append(h5_file['train_images'][idx]) for idx in rand_idx]
    [labels.append(LABELS[np.argmax(h5_file['train_labels'][idx])]) for idx in rand_idx]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for i in range(len(images)):
        s = plt.subplot(4, 6, i + 1)
        s.set_axis_off()
        plt.imshow(images[i])
        plt.title(labels[i])

    plt.show()


# package()
check_package()