# trainer.py
# Created by abdularis on 16/05/18

import tensorflow as tf
import cnn
from data_reader import DataSet


NUM_EPOCHS = 50
BATCH_SIZE = 128
DATASET_FILENAME = 'dataset/dataset.h5'
MODEL_FILENAME = 'aar.net'

model = cnn.build_model()
data_set = DataSet(DATASET_FILENAME, batch_size=BATCH_SIZE)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        for epoch in range(NUM_EPOCHS):
            for step in range(data_set.batch_count):
                batch_images, batch_labels = data_set.next_batch()
                train_accuracy, loss = model.train_step(sess, batch_images, batch_labels)
                print('Epoch %d, step %d, training accuracy: %f, training loss %f'
                      % (epoch, step, train_accuracy, loss))

            val_images, val_labels = data_set.val_data()
            validation_accuracy = model.evaluate(sess, val_images, val_labels)
            print('\tValidation accuracy: %f' % validation_accuracy)
            saver.save(sess, save_path=MODEL_FILENAME, global_step=epoch)
    except KeyboardInterrupt:
        print('Stop training.')
