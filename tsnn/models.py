# model.py
# Created by abdularis on 16/05/18

import tensorflow as tf
from .operations import Dropout

NOT_COMPILE_ERR_MSG = 'Model belum di compile: make sure you have call compile()'


class Model(object):

    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.x = tf.placeholder(tf.float32, shape=input_shape, name='input_placeholder')
        self.y = tf.placeholder(tf.float32, [None, num_classes], name='y_truth')
        self.model = self.x
        self.name_scope = None
        self.param_count = 0
        self.stored_ops = {}

        self.is_compiled = False
        self.loss = None
        self.optimizer = None
        self.prediction_accuracy = None
        self.prediction = None
        self.train_summary = None
        self.val_summary = None
        self.test_summary = None
        self.dropout_enabled = False
        self.dropout_keep_prob = 1.0

    def use_name_scope(self, name):
        self.name_scope = "%s/" % name

    def clear_name_scope(self):
        self.name_scope = None

    def _add_operation(self, operation):
        self.model = operation.get_operation_graph(self.model)
        if isinstance(operation, Dropout):
            self.dropout_enabled = True

    def add(self, operation, store_ref=False, name_ref=None):
        if self.is_compiled:
            return

        if self.name_scope:
            with tf.name_scope(self.name_scope):
                self._add_operation(operation)
        else:
            self._add_operation(operation)
        self.param_count += operation.param_count

        if store_ref:
            self.stored_ops[name_ref] = self.model

    def compile(self, optimizer, dropout_keep_prob=1.0):
        output_score = self.model

        if self.dropout_enabled:
            self.dropout_keep_prob = dropout_keep_prob

        # cross entropy loss
        with tf.name_scope('cross_entropy'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=output_score, labels=self.y)
            loss = tf.reduce_mean(loss)

        with tf.name_scope('train'):
            optimizer = optimizer.minimize(loss)

        with tf.name_scope('pred_accuracy'):
            y_truth = tf.argmax(self.y, 1)
            y_prediction = tf.argmax(output_score, 1)
            correct_prediction = tf.cast(tf.equal(y_prediction, y_truth), tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

        # for training
        self.loss = loss
        self.optimizer = optimizer
        self.prediction_accuracy = accuracy

        # prediction
        self.prediction = tf.nn.softmax(logits=output_score, name='inference')

        self.is_compiled = True

        # set summary
        self.train_summary = tf.summary.merge(
            [
                tf.summary.scalar('accuracy', self.prediction_accuracy),
                tf.summary.scalar('cross_entropy_loss', self.loss)
            ]
        )
        self.val_summary = tf.summary.merge(
            [
                tf.summary.scalar('val_accuracy', self.prediction_accuracy),
                tf.summary.scalar('val_cross_entropy_loss', self.loss)
            ]
        )
        self.test_summary = tf.summary.merge(
            [
                tf.summary.scalar('test_accuracy', self.prediction_accuracy)
            ]
        )

    def train_step(self, session, x, y, run_summary=False):
        if not self.is_compiled:
            raise ValueError(NOT_COMPILE_ERR_MSG)

        fetches = [self.prediction_accuracy, self.loss, self.optimizer]
        feed_dict = {self.x: x, self.y: y}
        if self.dropout_enabled:
            feed_dict[Dropout.keep_prob] = self.dropout_keep_prob

        if run_summary:
            fetches.append(self.train_summary)
            result = session.run(fetches=fetches, feed_dict=feed_dict)
            # return training accuracy, loss, summary
            return result[0], result[1], result[3]
        else:
            result = session.run(fetches=fetches, feed_dict=feed_dict)
            # return training accuracy, loss
            return result[0], result[1]

    def evaluate(self, session, x, y):
        if not self.is_compiled:
            raise ValueError(NOT_COMPILE_ERR_MSG)

        feed_dict = {self.x: x, self.y: y}
        if self.dropout_enabled:
            feed_dict[Dropout.keep_prob] = 1.0
        # return accuracy, loss, summary
        return session.run([self.prediction_accuracy, self.loss, self.val_summary], feed_dict=feed_dict)

    def predict(self, session, x, extra_fetches=None, extra_feed_dict=None):
        if not self.is_compiled:
            raise ValueError(NOT_COMPILE_ERR_MSG)
        fetches = [self.prediction]
        feed_dict = {self.x: x}
        if self.dropout_enabled:
            feed_dict[Dropout.keep_prob] = 1.0

        if extra_fetches:
            fetches.extend(extra_fetches)
        if extra_feed_dict:
            feed_dict = {**feed_dict, **extra_feed_dict}

        return session.run(fetches, feed_dict=feed_dict)
