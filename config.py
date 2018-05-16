# config.py
# Created by abdularis on 24/04/18

import os
import numpy as np


OUTPUT_IMAGE_SIZE = 136

BASE_ORIGINAL_DATA_DIR = 'dataset/original/'
BASE_OUTPUT_RESIZE_DIR = os.path.join('dataset/', str(OUTPUT_IMAGE_SIZE))
BASE_OUTPUT_AUG_DIR = os.path.join('dataset/', str(OUTPUT_IMAGE_SIZE) + 'aug')
CATEGORIES_DIR_NAME = [
    'backpack',
    'jeep',
    'microwave',
    'mug',
    'shoe',
    'teapot'
]

CATEGORIES_DIR_NAME_AND_AUG_COUNT = [
    ('backpack', 4),
    ('jeep', 3),
    ('microwave', 4),
    ('mug', 2),
    ('shoe', 2),
    ('teapot', 2)
]

# urutan ONE_HOT_IMAGE_LABELS dan LABELS harus sama
ONE_HOT_IMAGE_LABELS = {
    'backpack':  np.array([1., 0., 0., 0., 0., 0.]),
    'jeep':      np.array([0., 1., 0., 0., 0., 0.]),
    'microwave': np.array([0., 0., 1., 0., 0., 0.]),
    'mug':       np.array([0., 0., 0., 1., 0., 0.]),
    'shoe':      np.array([0., 0., 0., 0., 1., 0.]),
    'teapot':    np.array([0., 0., 0., 0., 0., 1.])
}

LABELS = [
    'backpack',
    'jeep',
    'microwave',
    'mug',
    'shoe',
    'teapot'
]

PKG_OUT_FILENAME = 'dataset.h5'
PKG_INPUT_BASE_DIR = BASE_OUTPUT_RESIZE_DIR

PKG_TRAIN_PERCENT = 80
PKG_VAL_PERCENT = 10
PKG_TEST_PERCENT = 10
