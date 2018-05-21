# trainer.py
# Created by abdularis on 16/05/18

import tensorflow as tf
import cnn
from data_reader import DataSet


INPUT_SHAPE = [None, 128, 128, 3]
NUM_CLASSES = 6
LEARNING_RATE = 1e-4

NUM_EPOCHS = 50
BATCH_SIZE = 64
DATASET_FILENAME = 'dataset/dataset.h5'
MODEL_FILENAME = './model/aar_net'
RUN_NAME = '1'

model = cnn.build_model_arch(INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE)
data_set = DataSet(DATASET_FILENAME, batch_size=BATCH_SIZE)

global_step = 0
file_writer = tf.summary.FileWriter('logs/%s/' % RUN_NAME)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_writer.add_graph(sess.graph)

    try:
        for epoch in range(NUM_EPOCHS):
            for step in range(data_set.batch_count):
                batch_images, batch_labels = data_set.next_batch()
                if step % 50 == 0:
                    train_accuracy, loss, summary = model.train_step(sess, batch_images, batch_labels, run_summary=True)
                    file_writer.add_summary(summary, global_step)
                else:
                    train_accuracy, loss = model.train_step(sess, batch_images, batch_labels, run_summary=False)
                    print('Epoch %d, step %d, global step %d, training accuracy: %f, training loss %f'
                          % (epoch, step, global_step, train_accuracy, loss))
                global_step += 1

            val_images, val_labels = data_set.val_data()
            val_accuracy, loss, summary = model.evaluate(sess, val_images, val_labels)
            print('\tValidation accuracy: %f, loss %f' % (val_accuracy, loss))
            file_writer.add_summary(summary, epoch)
            saver.save(sess, save_path=MODEL_FILENAME, global_step=epoch)
    except KeyboardInterrupt:
        print('Training cancelled intentionally.')

    print('Stop training at %d steps' % global_step)
    file_writer.close()

