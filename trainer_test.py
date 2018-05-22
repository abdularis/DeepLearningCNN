# trainer_test.py
# Created by abdularis on 22/05/18

import tensorflow as tf
import data_reader
import cnn
import numpy as np

model = cnn.build_model_arch([None, 128, 128, 3], 6, 1e-3)
_, _, test_data = data_reader.read_data_set('dataset/dataset.h5', 64)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'model/aar_net-2-48')

    accuracies = []
    for step in range(test_data.batch_count):
        images, labels = test_data.next_batch()
        images = images / 255.0

        pred = model.predict(sess, images)
        pred = np.argmax(pred, 1)

        truth = np.argmax(labels, 1)

        acc = (pred == truth).astype(np.float32).mean()
        print("Step %d, accuracy %f" % (step, acc))
        accuracies.append(acc)

    print("Overall accuracy: %f" % np.mean(accuracies))
