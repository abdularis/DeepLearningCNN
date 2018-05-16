# resizer.py
# Created by abdularis on 27/03/18

import scipy.misc
import os
import ntpath
import argparse
from tqdm import tqdm
from skimage import transform


def resize_images(images_dir, output_dir, img_size):
    if not os.path.exists(images_dir):
        print('Direktori %s tidak ada' % images_dir)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('[*] Memulai resize image dari "%s" --ke-> "%s", output %dx%d pixel' %
          (images_dir, output_dir, img_size[0], img_size[1]))
    for file_path in tqdm(os.listdir(images_dir)):
        file_path = os.path.join(images_dir, file_path)

        if not os.path.isfile(file_path):
            print('[!] "%s" adalah direktori, abaikan' % file_path)
            continue

        file_name = ntpath.split(file_path)[-1]
        if file_name == '':
            return

        try:
            img = scipy.misc.imread(file_path, mode='RGB')
            img = transform.resize(img, (img_size[0], img_size[1]), mode='constant')

            file_name = os.path.splitext(file_name)[0] + '.jpg'
            scipy.misc.imsave(os.path.join(output_dir, file_name), img, format='jpeg')
        except Exception as msg:
            print("[!] Gagal resize %s, melanjutkan" % file_path)
            print("[!] Exception Message: %s" % msg)
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize image')
    parser.add_argument('--image-dir', type=str, help='Direktori tempat image berada', required=True)
    parser.add_argument('--output-dir', type=str, help='Direktori output image', required=True)
    parser.add_argument('--image-size', type=str, help='Ukuran output image (cth: 128x128)', required=True)

    args = parser.parse_args()

    image_size = [int(i) for i in args.image_size.split('x')]
    if len(image_size) < 2:
        print('image size error')
    else:
        resize_images(args.image_dir, args.output_dir, image_size)
