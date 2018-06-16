# splitter.py
# Created by abdularis on 04/06/18

import json
import argparse
import glob
import os
import random
import shutil
import tqdm


def _move_files(out_dir, folder_name, train_paths, val_paths, test_paths):
    train_folder = os.path.join(out_dir, 'train/', folder_name)
    val_folder = os.path.join(out_dir, 'validation/', folder_name)
    test_folder = os.path.join(out_dir, 'test/', folder_name)

    os.makedirs(train_folder)
    os.makedirs(val_folder)
    os.makedirs(test_folder)

    print('move from %s --> %s' % (folder_name, train_folder))
    for path in tqdm.tqdm(train_paths): shutil.copy2(path, train_folder)
    print('move from %s --> %s' % (folder_name, val_folder))
    for path in tqdm.tqdm(val_paths): shutil.copy2(path, val_folder)
    print('move from %s --> %s' % (folder_name, test_folder))
    for path in tqdm.tqdm(test_paths): shutil.copy2(path, test_folder)


def split(base_dir, folder_names, out_dir, train_percent, val_percent, test_percent):
    if (train_percent + val_percent + test_percent) != 100:
        print('Train % + validation % + test % != 100%')
        return

    for folder in folder_names:
        folder_path = os.path.join(base_dir, folder)
        paths = glob.glob(os.path.join(folder_path, '*'))
        random.shuffle(paths)

        train_offset = round(len(paths) / 100 * train_percent)
        valid_offset = round(len(paths) / 100 * (train_percent + val_percent))
        train_paths, val_paths, test_paths =\
            paths[:train_offset], paths[train_offset:valid_offset], paths[valid_offset:]

        _move_files(out_dir, folder, train_paths, val_paths, test_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data splitter')
    parser.add_argument('-c', dest='cfg_file', help='Path to configuration file', required=True)

    args = parser.parse_args()

    cfg = json.load(open(args.cfg_file, 'r'))
    split(cfg['base_dir'],
          cfg['folder_names'],
          cfg['output_dir'],
          cfg['train_split'],
          cfg['val_split'],
          cfg['test_split'])
