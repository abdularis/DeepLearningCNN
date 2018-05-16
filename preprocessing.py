# run_preprocessing.py
# Created by abdularis on 29/03/18

import argparse
import os
import config as cfg


def do_resize_all_images():
    from preprocess.resizer import resize_images

    for dir_name in cfg.CATEGORIES_DIR_NAME:
        resize_images(images_dir=os.path.join(cfg.BASE_ORIGINAL_DATA_DIR, dir_name),
                      output_dir=os.path.join(cfg.BASE_OUTPUT_RESIZE_DIR, dir_name),
                      img_size=cfg.OUTPUT_IMAGE_SIZE)


def do_augment_all_images():
    from preprocess.augmenter import augment_images

    for dir_name, augment_count in cfg.CATEGORIES_DIR_NAME_AND_AUG_COUNT:
        if augment_count <= 0:
            continue
        augment_images(images_dir=os.path.join(cfg.BASE_OUTPUT_RESIZE_DIR, dir_name),
                       output_dir=os.path.join(cfg.BASE_OUTPUT_AUG_DIR, dir_name),
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
