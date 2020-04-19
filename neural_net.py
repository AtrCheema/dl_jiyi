__all__ = ["Model"]

from tensorflow.keras.layers import LSTMCell, Dropout, LSTM, BatchNormalization, Activation  # , CuDNNLSTM
from tensorflow.python.keras.layers import RNN
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops.rnn import dynamic_rnn
from TSErrors import FindErrors

#from utils import check_min_loss, batch_generator, Batch_Generator
from leaky_dense import LeakyDense2D

import numpy as np
import os
import time


def reset_graph(seed=2):
    # tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()
    # tf.set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    # np.random.seed(seed)


class Model(object):

    def __init__(self, nn_config, verbose=1):
        self.nn_conf = nn_config
        self.x_ph = None
        self.mask_ph = None
        self.obs_y_ph = None
        self.full_outputs = None
        self.training_op = None
        self.loss = None
        self.saver = None
        self.config = {}
        self.verbose = verbose


    def build_nn(self):

        reset_graph()

        n_steps = self.nn_conf['lookback']

        self.x_ph = tf.compat.v1.placeholder(tf.float32, [None, n_steps, self.nn_conf['input_features']],
                                             name='x_ph')  # self.nn_conf['batch_size']
        self.obs_y_ph = tf.compat.v1.placeholder(tf.float32, [None, 1], name='obs_y_ph')
        self.mask_ph = tf.compat.v1.placeholder(tf.float32, [None, 1], name='mask_ph')  # self.nn_conf['batch_size']

        cell = LSTMCell(units=self.nn_conf['lstm_units'], activation=self.nn_conf['lstm_activation'])

        if self.nn_conf['method'] == 'dynamic_rnn':
            rnn_outputs1, states = dynamic_rnn(cell, self.x_ph, dtype=tf.float32)
            rnn_outputs = tf.reshape(rnn_outputs1[:, -1, :], [-1, self.nn_conf['lstm_units']])

        elif self.nn_conf['method'] == 'keras_lstm_layer':
            rnn_outputs = LSTM(self.nn_conf['lstm_units'],
                               activation=self.nn_conf['lstm_activation'],
                               input_shape=(n_steps, self.nn_conf['input_features']))(self.x_ph)

        else:
            rnn_layer = RNN(cell)
            rnn_outputs = rnn_layer(self.x_ph)  # [batch_size, neurons]
            if self.verbose > 0: print(rnn_outputs.shape, 'before reshaping', K.eval(tf.rank(rnn_outputs)))
            rnn_outputs = tf.reshape(rnn_outputs[:, :], [-1, self.nn_conf['lstm_units']])
            if self.verbose > 0: print(rnn_outputs.shape, 'after reshaping', K.eval(tf.rank(rnn_outputs)))

        if self.nn_conf['batch_norm']:
            rnn_outputs3 = BatchNormalization()(rnn_outputs)
            rnn_outputs = Activation('relu')(rnn_outputs)

        if self.nn_conf['dropout'] is not None:
            rnn_outputs = Dropout(self.nn_conf['dropout'])(rnn_outputs)

        if self.verbose > 0: print(rnn_outputs.shape, "dynamic_rnn/RNN outputs shape", rnn_outputs.get_shape())

        leaky_layer = LeakyDense2D(units=self.nn_conf['output_features'], activation=tf.nn.elu, leaky_inputs=True,
                                   mask_array=self.mask_ph, verbose=self.verbose)  # tf.nn.elu
        dense_outputs = leaky_layer(rnn_outputs)

        self.nn_conf['dense_activation'] = leaky_layer.activation.__name__

        outputs = tf.reshape(dense_outputs, [-1, 1])  # this is because observations will also be flattened
        self.full_outputs = leaky_layer.full_outputs

        if self.verbose > 0: print(outputs.shape, 'shape outputs')

        self.loss = tf.reduce_mean(tf.square(outputs - self.obs_y_ph))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.nn_conf['lr'])
        self.training_op = optimizer.minimize(self.loss)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.nn_conf['n_epochs'])