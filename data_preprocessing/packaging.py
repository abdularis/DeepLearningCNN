# packaging.py
# Created by abdularis on 30/03/18

import os
import random
import argparse
import scipy.misc
import h5py
import numpy as np
from tqdm import tqdm
import config as cfg


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

    for category_name in cfg.ONE_HOT_IMAGE_LABELS:
        dir_path = os.path.join(cfg.PKG_INPUT_BASE_DIR, category_name)

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

    for category_name in cfg.ONE_HOT_IMAGE_LABELS:
        for path in train[category_name]:
            train_paths.append(path)
            train_labels.append(cfg.ONE_HOT_IMAGE_LABELS[category_name])
        for path in validation[category_name]:
            val_paths.append(path)
            val_labels.append(cfg.ONE_HOT_IMAGE_LABELS[category_name])
        for path in test[category_name]:
            test_paths.append(path)
            test_labels.append(cfg.ONE_HOT_IMAGE_LABELS[category_name])

    train_paths, train_labels = shuffle(train_paths, train_labels)
    val_paths, val_labels = shuffle(val_paths, val_labels)
    test_paths, test_labels = shuffle(test_paths, test_labels)

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def package():
    print('[*] Packaging...')

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = get_dataset_paths(
        cfg.PKG_TRAIN_PERCENT, cfg.PKG_VAL_PERCENT, cfg.PKG_TEST_PERCENT)

    train_shape = (len(train_paths), cfg.OUTPUT_IMAGE_SIZE, cfg.OUTPUT_IMAGE_SIZE, 3)
    val_shape = (len(val_paths), cfg.OUTPUT_IMAGE_SIZE, cfg.OUTPUT_IMAGE_SIZE, 3)
    test_shape = (len(test_paths), cfg.OUTPUT_IMAGE_SIZE, cfg.OUTPUT_IMAGE_SIZE, 3)

    h5_file = h5py.File(cfg.PKG_OUT_FILENAME, mode='w')
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


def check_package(dataset_name, label_names):
    if not os.path.exists(cfg.PKG_OUT_FILENAME):
        print('File %s tidak ada.' % cfg.PKG_OUT_FILENAME)
        return

    h5_file = h5py.File(cfg.PKG_OUT_FILENAME, mode='r')

    rand_idx = np.random.random_integers(0, len(h5_file[dataset_name]), 24)
    images = []
    labels = []
    [images.append(h5_file[dataset_name][idx]) for idx in rand_idx]
    [labels.append(cfg.LABELS[np.argmax(h5_file[label_names][idx])]) for idx in rand_idx]

    print('Images : %d, Labels: %d' % (len(h5_file[dataset_name]), len(h5_file[label_names])))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for i in range(len(images)):
        s = plt.subplot(4, 6, i + 1)
        s.set_axis_off()
        plt.imshow(images[i])
        plt.title(labels[i])

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--run', dest='run', help='Package atau check package ["package" or "check"]')

    args = parser.parse_args()
    if args.run == 'package':
        package()
    elif args.run == 'check_train':
        check_package('train_images', 'train_labels')
    elif args.run == 'check_val':
        check_package('val_images', 'val_labels')
    elif args.run == 'check_test':
        check_package('test_images', 'test_labels')
