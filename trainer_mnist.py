# trainer_mnist.py
# Created by abdularis on 17/05/18

import cnn
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


NUM_EPOCHS = 50
BATCH_SIZE = 64
MODEL_FILENAME = './model/mnist.ckpt'
RUN_NAME = '1'


data_set = input_data.read_data_sets('./dataset/MNIST_data', one_hot=True)

model = cnn.build_model_arch([None, 28, 28, 1], 10, 1e-4)

global_step = 0
file_writer = tf.summary.FileWriter('logs/%s/' % RUN_NAME)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_writer.add_graph(sess.graph)

    try:
        mini_batch_count = int(data_set.train.num_examples / BATCH_SIZE)
        for epoch in range(NUM_EPOCHS):
            for step in range(mini_batch_count):
                batch_images, batch_labels = data_set.train.next_batch(BATCH_SIZE)
                batch_images = batch_images.reshape((batch_images.shape[0], 28, 28, 1))

                if step % 100 == 0:
                    train_accuracy, loss = model.train_step(sess, batch_images, batch_labels, run_summary=True)
                else:
                    train_accuracy, loss, summary = model.train_step(sess, batch_images, batch_labels, run_summary=True)
                    file_writer.add_summary(summary, global_step)
                print('Epoch %d, step %d, global step %d, training accuracy: %f, training loss %f'
                      % (epoch, step, global_step, train_accuracy, loss))
                global_step += 1

            # validation
            for step in range(int(data_set.validation.num_examples / BATCH_SIZE)):
                batch_images, batch_labels = data_set.train.next_batch(BATCH_SIZE)
                batch_images = batch_images.reshape((batch_images.shape[0], 28, 28, 1))

                val_accuracy, loss, summary = model.evaluate(sess, batch_images, batch_labels)
                print('\tStep %d: validation accuracy %f, loss %f'
                      % (step, val_accuracy, loss))
                file_writer.add_summary(summary, step)

            print('save')
            saver.save(sess, save_path=MODEL_FILENAME, global_step=epoch)

    except KeyboardInterrupt:
        print('Training cancelled intentionally.')

    print('Stop training at %d steps' % global_step)
    file_writer.close()
