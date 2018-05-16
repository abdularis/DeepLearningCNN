# packaging.py
# Created by abdularis on 30/03/18

import os
import json
import random
import argparse
import scipy.misc
import h5py
import numpy as np
from tqdm import tqdm


def shuffle(list1, list2):
    temp_list = list(zip(list1, list2))
    random.shuffle(temp_list)
    return zip(*temp_list)


def get_dataset_split_paths(base_dir, train_percent, validation_percent, test_percent):
    if (train_percent + validation_percent + test_percent) != 100:
        print('Train % + validation % + test % != 100%')
        return

    train = {}
    validation = {}
    test = {}

    one_hot = create_one_hot_from_dir(base_dir)
    directories = os.listdir(base_dir)
    for category_name in directories:
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

    for category_name in one_hot:
        for path in train[category_name]:
            train_paths.append(path)
            train_labels.append(one_hot[category_name])
        for path in validation[category_name]:
            val_paths.append(path)
            val_labels.append(one_hot[category_name])
        for path in test[category_name]:
            test_paths.append(path)
            test_labels.append(one_hot[category_name])

    train_paths, train_labels = shuffle(train_paths, train_labels)
    val_paths, val_labels = shuffle(val_paths, val_labels)
    test_paths, test_labels = shuffle(test_paths, test_labels)

    return one_hot, (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def create_one_hot_from_dir(base_dir):
    one_hot = {}
    dirs = os.listdir(base_dir)
    dir_count = len(dirs)
    for i in range(dir_count):
        encoding = np.zeros(dir_count, dtype=np.float32)
        encoding[i] = 1.0  # one hot
        one_hot[dirs[i]] = encoding
    return one_hot


def package(base_dir, img_size, out_filename, train_percent, val_percent, test_percent):
    print('[*] Packaging...')

    one_hot, (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) =\
        get_dataset_split_paths(base_dir, train_percent, val_percent, test_percent)

    train_shape = (len(train_paths), img_size[0], img_size[1], img_size[2])
    val_shape = (len(val_paths), img_size[0], img_size[1], img_size[2])
    test_shape = (len(test_paths), img_size[0], img_size[1], img_size[2])

    h5_file = h5py.File(out_filename, mode='w')
    h5_file.create_dataset('train_images', train_shape, np.uint8)
    h5_file.create_dataset('train_labels', data=train_labels)
    h5_file.create_dataset('val_images', val_shape, np.uint8)
    h5_file.create_dataset('val_labels', data=val_labels)
    h5_file.create_dataset('test_images', test_shape, np.uint8)
    h5_file.create_dataset('test_labels', data=test_labels)

    # create dataset group for one hot encoding
    [h5_file.create_dataset('label_encoding/%s' % lbl, data=one_hot[lbl]) for lbl in one_hot]

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


def check_package(filename, dataset_name, label_names):
    if not os.path.exists(filename):
        print('File %s tidak ada.' % filename)
        return

    h5_file = h5py.File(filename, mode='r')
    encoded_labels = {}
    for lbl in h5_file['label_encoding']:
        idx = np.argmax(h5_file['label_encoding'][lbl])
        encoded_labels[idx] = lbl

    rand_idx = np.random.random_integers(0, len(h5_file[dataset_name]), 24)
    images = []
    labels = []
    [images.append(h5_file[dataset_name][idx]) for idx in rand_idx]
    [labels.append(encoded_labels[int(np.argmax(h5_file[label_names][idx]))]) for idx in rand_idx]

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
    parser = argparse.ArgumentParser(description='Data packaging')
    parser.add_argument('-c', dest='cfg_file', help='Path to configuration file', required=True)
    parser.add_argument('--pack', action='store_true', help='Pack data set')
    parser.add_argument('--check-train', action='store_true', help='Show training images')
    parser.add_argument('--check-validation', action='store_true', help='Show validation images')
    parser.add_argument('--check-test', action='store_true', help='Show test images')

    args = parser.parse_args()

    cfg = json.load(open(args.cfg_file, 'r'))
    if args.pack:
        img_size = [int(i) for i in cfg['img_size'].split('x')]
        package(cfg['base_dir'],
                img_size,
                cfg['filename'],
                cfg['train'],
                cfg['validation'],
                cfg['test'])
    elif args.check_train:
        check_package(cfg['filename'], 'train_images', 'train_labels')
    elif args.check_validation:
        check_package(cfg['filename'], 'val_images', 'val_labels')
    elif args.check_test:
        check_package(cfg['filename'], 'test_images', 'test_labels')
