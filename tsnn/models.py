# model.py
# Created by abdularis on 16/05/18

import tensorflow as tf

NOT_COMPILE_ERR_MSG = 'Model belum di compile: make sure you have call compile()'


class Model(object):

    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.x = tf.placeholder(tf.float32, shape=input_shape, name='input_placeholder')
        self.y = tf.placeholder(tf.float32, [None, num_classes], name='y_truth')
        self.model = self.x
        self.name_scope = None
        self.param_count = 0

        self.is_compiled = False
        self.loss = None
        self.optimizer = None
        self.prediction_accuracy = None
        self.prediction = None

    def use_name_scope(self, name):
        self.name_scope = "%s/" % name

    def clear_name_scope(self):
        self.name_scope = None

    def add(self, operation):
        if self.is_compiled:
            return

        if self.name_scope:
            with tf.name_scope(self.name_scope):
                self.model = operation.get_operation_graph(self.model)
        else:
            self.model = operation.get_operation_graph(self.model)
        self.param_count += operation.param_count

    def compile(self):
        output_score = self.model

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=output_score, labels=self.y)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss)

        y_truth = tf.argmax(self.y, 1)
        y_prediction = tf.argmax(output_score, 1)
        correct_prediction = tf.cast(tf.equal(y_prediction, y_truth), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        # for training
        self.loss = loss
        self.optimizer = optimizer
        self.prediction_accuracy = accuracy

        # prediction
        self.prediction = tf.nn.softmax(logits=output_score, name='output_prediction')

        self.is_compiled = True

    def train_step(self, session, x, y):
        if not self.is_compiled:
            raise ValueError(NOT_COMPILE_ERR_MSG)
        train_accuracy, loss, _ = session.run([self.prediction_accuracy, self.loss, self.optimizer], feed_dict={self.x: x, self.y: y})
        return train_accuracy, loss

    def evaluate(self, session, x, y):
        if not self.is_compiled:
            raise ValueError(NOT_COMPILE_ERR_MSG)
        eval_accuracy = session.run(self.prediction_accuracy, feed_dict={self.x: x, self.y: y})
        return eval_accuracy

    def predict(self, session, x):
        if not self.is_compiled:
            raise ValueError(NOT_COMPILE_ERR_MSG)
        return session.run(self.prediction, feed_dict={self.x: x})
