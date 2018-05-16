# augmenter.py
# Created by abdularis on 26/03/18

import scipy.misc
import numpy as np
import os
import argparse
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator


# augmentasi data citra pada direktori 'image_dir' output ke 'output_dir'
# dengan jumlah augmentasi percitra 'augment_per_image'
def augment_images(image_dir, output_dir, augment_per_image):
    if not os.path.exists(image_dir):
        print('Direktori %s tidak ada' % image_dir)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, horizontal_flip=True)
    for file_name in tqdm(os.listdir(image_dir)):
        file_path = os.path.join(image_dir, file_name)

        if not os.path.isfile(file_path):
            print('[!] "%s" adalah direktori, abaikan' % file_path)
            continue

        img = np.array(scipy.misc.imread(file_path, mode='RGB'), ndmin=4)
        data_gen.fit(img)

        scipy.misc.imsave(os.path.join(output_dir, file_name), img[0])
        count = 0
        for _ in data_gen.flow(img, None, batch_size=1, save_to_dir=output_dir, save_format='jpg', save_prefix='aug'):
            count += 1
            if count >= augment_per_image:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize image')
    parser.add_argument('--image-dir', type=str, help='Direktori tempat image yang akan diaugmentasi', required=True)
    parser.add_argument('--output-dir', type=str, help='Direktori output image', required=True)
    parser.add_argument('--augment-per-image', type=int, help='Jumlah augmentasi per image', required=True)

    args = parser.parse_args()
    augment_images(args.image_dir, args.output_dir, args.augment_per_image)
