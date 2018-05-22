# cnn.py
# Created by abdularis on 24/04/18


import tensorflow as tf
from tsnn.models import Model
from tsnn.operations import Convolution, Relu, MaxPooling, Flatten, FullyConnected


def build_model_arch(input_shape, num_classes, learning_rate):
    # model construction
    model = Model(input_shape=input_shape, num_classes=num_classes)
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
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_6')
    model.add(Convolution(128, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('flatten')
    model.add(Flatten())

    model.use_name_scope('fully_connected_1')
    model.add(FullyConnected(512))
    model.add(Relu())

    model.use_name_scope('fully_connected_2')
    model.add(FullyConnected(512))
    model.add(Relu())

    model.use_name_scope('fully_connected_3')
    model.add(FullyConnected(num_classes))
    # end

    model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate))
    return model


def build_model_arch_v2(input_shape, num_classes, learning_rate):
    # model construction
    model = Model(input_shape=input_shape, num_classes=num_classes)
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

    model.use_name_scope('conv_4')
    model.add(Convolution(64, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_5')
    model.add(Convolution(128, (3, 3)))
    model.add(Relu())

    model.use_name_scope('conv_6')
    model.add(Convolution(128, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_7')
    model.add(Convolution(128, (3, 3)))
    model.add(Relu())

    model.use_name_scope('conv_8')
    model.add(Convolution(128, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_9')
    model.add(Convolution(256, (3, 3)))
    model.add(Relu())

    model.use_name_scope('conv_10')
    model.add(Convolution(256, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_11')
    model.add(Convolution(256, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('flatten')
    model.add(Flatten())

    model.use_name_scope('fully_connected_1')
    model.add(FullyConnected(512))
    model.add(Relu())

    model.use_name_scope('fully_connected_2')
    model.add(FullyConnected(1024))
    model.add(Relu())

    model.use_name_scope('fully_connected_3')
    model.add(FullyConnected(num_classes))
    # end

    model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate))
    return model
