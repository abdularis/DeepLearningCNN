# data_visualizer.py
# Created by abdularis on 08/06/18

import matplotlib.pyplot as plt
import numpy as np


def show_images(images, one_hot, one_hot_labels):
    # max 24 images shown
    images, one_hot = images[:24], one_hot[:24]

    labels = []
    for l in one_hot:
        lbl_idx = np.argmax(l)
        labels.append(one_hot_labels[lbl_idx])

    print('Images : %d, Labels: %d' % (len(images), len(one_hot)))

    plt.figure(figsize=(10, 6))
    for i in range(len(images)):
        s = plt.subplot(4, 6, i + 1)
        s.set_axis_off()
        plt.imshow(images[i])
        plt.title(labels[i])

    plt.show()


def show_images_with_truth(images, labels, truth, pred):
    # max 24 images shown
    images = images[:24]

    plt.figure(figsize=(10, 6))
    for i in range(len(images)):
        s = plt.subplot(4, 6, i + 1)
        s.set_axis_off()
        plt.imshow(images[i])
        t = plt.title(labels[i])
        if truth[i] != pred[i]:
            plt.setp(t, color='r')
    plt.show()
