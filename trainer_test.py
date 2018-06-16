# trainer_test.py
# Created by abdularis on 22/05/18

import argparse
import tensorflow as tf
import data_reader
import cnn
import numpy as np
import trainer as mcfg


def run_test_visual(dataset_dir, model_path):
    import data_visualizer as dv
    import data_config as cfg

    model = cnn.build_model_arch(mcfg.INPUT_SHAPE, mcfg.NUM_CLASSES, 0)
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


def run_test(dataset_dir, model_path, top_k=1):
    import data_config as cfg

    model = cnn.build_model_arch(mcfg.INPUT_SHAPE, mcfg.NUM_CLASSES, 0)
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


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Test model CNN')
#     parser.add_argument('--dataset-dir', type=int, help='Direktori dataset', required=True)
#     parser.add_argument('--model-path', type=int, help='Path model CNN', required=True)
#     parser.add_argument('--visual', type=bool, help='Show visual', required=False)
#
#