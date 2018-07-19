# cnnarch1_do2.py
# Created by abdularis on 05/07/18


import tensorflow as tf
from tsnn.models import Model
from tsnn.operations import Convolution, Relu, MaxPooling, Flatten, FullyConnected, Dropout


def build_model_arch():

    INPUT_SHAPE = [None, 128, 128, 3]
    NUM_CLASSES = 6
    LEARNING_RATE = 1e-4
    KEEP_PROB = 0.5

    # model construction
    model = Model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.use_name_scope('conv_1')
    model.add(Convolution(64, (3, 3)))
    model.add(Relu())

    model.use_name_scope('conv_2')
    model.add(Convolution(64, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_3')
    model.add(Convolution(64, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_4')
    model.add(Convolution(128, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_5')
    model.add(Convolution(128, (3, 3)))
    model.add(Relu(), store_ref=True, name_ref='features-cn5')
    model.add(MaxPooling())

    model.use_name_scope('conv_6')
    model.add(Convolution(128, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('flatten')
    model.add(Flatten(), store_ref=True, name_ref='features-flt')

    model.use_name_scope('fully_connected_1')
    model.add(FullyConnected(512))
    model.add(Relu(), store_ref=True, name_ref='features-fc1')
    model.add(Dropout())

    model.use_name_scope('fully_connected_2')
    model.add(FullyConnected(512))
    model.add(Relu(), store_ref=True, name_ref='features-fc2')
    model.add(Dropout())

    model.use_name_scope('fully_connected_3')
    model.add(FullyConnected(NUM_CLASSES))
    # end

    model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE), dropout_keep_prob=KEEP_PROB)
    return model
