# run_preprocessing.py
# Created by abdularis on 29/03/18

import argparse
import os


OUTPUT_IMAGE_SIZE = 160

BASE_ORIGINAL_DATA_DIR = 'dataset/original/'
BASE_OUTPUT_RESIZE_DIR = os.path.join('dataset/', str(OUTPUT_IMAGE_SIZE))
BASE_OUTPUT_AUG_DIR = os.path.join('dataset/', str(OUTPUT_IMAGE_SIZE) + 'aug')

DATA_DIR_NAMES = [('backpack', 4), ('cat', 0), ('jeep', 3), ('microwave', 4), ('mug', 2), ('teapot', 2)]


def do_resize_all_images():
    from resizer import resize_images

    for dir_name, _ in DATA_DIR_NAMES:
        resize_images(images_dir=os.path.join(BASE_ORIGINAL_DATA_DIR, dir_name),
                      output_dir=os.path.join(BASE_OUTPUT_RESIZE_DIR, dir_name),
                      img_size=OUTPUT_IMAGE_SIZE)


def do_augment_all_images():
    from augmenter import augment_images

    for dir_name, augment_count in DATA_DIR_NAMES:
        if augment_count <= 0:
            continue
        augment_images(images_dir=os.path.join(BASE_OUTPUT_RESIZE_DIR, dir_name),
                       output_dir=os.path.join(BASE_OUTPUT_AUG_DIR, dir_name),
                       augment_per_image=augment_count)


# Ini merupakan program entri point untuk preprocesing citra
# seperti resize, augmentasi dll
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--run', dest='run', help='Jenis preprosesing yang akan dijalankan, ["resize", "augment"]')

    args = parser.parse_args()
    if args.run == 'resize':
        do_resize_all_images()
    elif args.run == 'augment':
        do_augment_all_images()
    elif args.run == 'all':
        do_resize_all_images()
        do_augment_all_images()
    else:
        print('Silakan masukan preprocessing yang akan dijalankan')
