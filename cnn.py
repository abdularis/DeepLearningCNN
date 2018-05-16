# cnn.py
# Created by abdularis on 24/04/18

import tensorflow as tf
from tsnn.models import Model
from tsnn.operations import Convolution, Relu, MaxPooling, Flatten, FullyConnected
import data_reader


# model construction
model = Model(input_shape=[None, 136, 136, 3], num_classes=6)
model.use_name_scope('conv_1')
model.add(Convolution(64, (3, 3)))
model.add(Relu())

model.use_name_scope('conv_2')
model.add(Convolution(64, (3, 3)))
model.add(Relu())
model.add(MaxPooling())

model.use_name_scope('conv_5')
model.add(Convolution(64, (3, 3)))
model.add(Relu())
model.add(MaxPooling())

model.use_name_scope('conv_6')
model.add(Convolution(128, (3, 3)))
model.add(Relu())
model.add(MaxPooling())

model.use_name_scope('conv_7')
model.add(Convolution(128, (3, 3)))
model.add(Relu())
model.add(MaxPooling())

model.use_name_scope('conv_8')
model.add(Convolution(128, (3, 3)))
model.add(Relu())
model.add(MaxPooling())

model.use_name_scope('flatten')
model.add(Flatten())

model.use_name_scope('fully_connected_1')
model.add(FullyConnected(256))
model.add(Relu())

model.use_name_scope('fully_connected_2')
model.add(FullyConnected(512))
model.add(Relu())

model.use_name_scope('fully_connected_3')
model.add(FullyConnected(6))
#

dataset = data_reader.DataSet('dataset/dataset.h5', batch_size=20)
epochs = 50

model.compile()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_images, batch_labels = dataset.next_batch()

    try:
        for epoch in range(epochs):
            train_acc, loss = model.train_step(sess, batch_images, batch_labels)
            print('Epoch %d, training accuracy: %f, %f' % (epoch, train_acc, loss))
    except KeyboardInterrupt:
        print('Stopped.')