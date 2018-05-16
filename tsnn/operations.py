# operations.py
# Created by abdularis on 16/05/18

import tensorflow as tf
import numpy as np


def _create_weight(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def _create_bias(size, name=None):
    return tf.Variable(tf.constant(0.1, shape=[size]), name=name)


def _calc_param_count(param_shape):
    return int(np.prod(param_shape))


class BaseOperation(object):

    def __init__(self):
        self.param_count = 0

    def get_operation_graph(self, input):
        pass


class Convolution(BaseOperation):

    def __init__(self, num_filters, filter_size, strides=(1, 1, 1, 1)):
        super().__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides

    def get_operation_graph(self, input):
        weights = _create_weight([self.filter_size[0], self.filter_size[1], int(input.get_shape()[-1]), self.num_filters],
                                 name='weights')
        biases = _create_bias(self.num_filters, name='biases')

        layer = tf.nn.conv2d(input, weights, strides=list(self.strides), padding='SAME', name='conv')
        layer += biases

        self.param_count = _calc_param_count(weights.shape) + _calc_param_count(biases.shape)
        return layer


class Relu(BaseOperation):

    def get_operation_graph(self, input):
        return tf.nn.relu(input, name='activation')


class MaxPooling(BaseOperation):

    def __init__(self, kern_size=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
        super().__init__()
        self.kern_size = kern_size
        self.strides = strides

    def get_operation_graph(self, input):
        return tf.nn.max_pool(input, ksize=self.kern_size, strides=self.strides, padding='SAME', name='max_pool')


class Flatten(BaseOperation):

    def get_operation_graph(self, input):
        input_shape = input.get_shape()
        num_features = input_shape[1:4].num_elements()
        return tf.reshape(input, [-1, num_features])


class FullyConnected(BaseOperation):

    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def get_operation_graph(self, input):
        weights = _create_weight([int(input.get_shape()[-1]), self.num_neurons], name='weights')
        biases = _create_bias(self.num_neurons, name='biases')

        layer = tf.matmul(input, weights) + biases

        self.param_count = _calc_param_count(weights.shape) + _calc_param_count(biases.shape)
        return layer
