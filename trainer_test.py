# trainer_test.py
# Created by abdularis on 22/05/18

import argparse
import importlib

import tensorflow as tf
import data_reader
import numpy as np


def run_test_visual(model_arch_module, dataset_dir, model_path):
    import data_visualizer as dv
    import data_config as cfg

    model = model_arch_module.build_model_arch()
    _, _, test_data = data_reader.read_data_set_dir(dataset_dir, cfg.one_hot, 64)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        labels = cfg.one_hot_labels

        for step in range(test_data.batch_count):
            images, one_hot = test_data.next_batch()
            images = images / 255.0
            truth_indexes = np.argmax(one_hot, 1)

            pred = model.predict(sess, images)
            pred_indexes = np.argmax(pred, 1)
            pred_labels = [labels[i] for i in pred_indexes]

            dv.show_images_with_truth(images, pred_labels, truth_indexes, pred_indexes)


def run_test(model_arch_module, dataset_dir, model_path, top_k=1):
    import data_config as cfg

    model = model_arch_module.build_model_arch()
    _, _, test_data = data_reader.read_data_set_dir(dataset_dir, cfg.one_hot, 64)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        accuracies = []
        for step in range(test_data.batch_count):
            images, one_hot = test_data.next_batch()
            images = images / 255.0
            truth_indexes = np.argmax(one_hot, 1)

            pred = model.predict(sess, images)
            pred_indexes = np.argmax(pred, 1)

            if top_k == 1:
                acc = (truth_indexes == pred_indexes).astype(np.float32).mean()
            else:
                acc = tf.nn.in_top_k(pred_indexes, truth_indexes, k=top_k)
                acc = sess.run(acc).astype(np.float32).mean()

            accuracies.append(acc)
            print("Step %d, accuracy %f" % (step, acc))

        overall_acc = np.mean(accuracies)
        print("Overall accuracy: %f" % overall_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model CNN')
    parser.add_argument('--model-module', type=str, help='Python module string untuk model cnn', required=True)
    parser.add_argument('--dataset-dir', type=str, help='Direktori dataset', required=True)
    parser.add_argument('--model-path', type=str, help='Path model CNN', required=True)
    parser.add_argument('--type', type=str, help='Jalankan test visual atau cmd [cmd | vis]', default='cmd', required=False)

    args = parser.parse_args()

    print('Run trainer:')
    print('\tModel module name: %s' % args.model_module)
    print('\tDataset dir: %s' % args.dataset_dir)
    print('\tModel path: %s' % args.model_path)
    if args.type == 'vis':
        print('================= Visualize =================')
        run_test_visual(importlib.import_module(args.model_module), args.dataset_dir, args.model_path)
    elif args.type == 'cmd':
        print('Run test')
        run_test(importlib.import_module(args.model_module), args.dataset_dir, args.model_path)
