__all__ = ["NeuralNetwork"]

from tensorflow.keras.layers import LSTMCell, Dropout, LSTM, BatchNormalization, Activation  # , CuDNNLSTM
from tensorflow.python.keras.layers import RNN
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops.rnn import dynamic_rnn
from TSErrors import FindErrors

from utils import check_min_loss  # , Batch_Generator
from leaky_dense import LeakyDense2D

import numpy as np
import os
import time
from collections import OrderedDict


def reset_graph(seed=2):
    # tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()
    # tf.set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    # np.random.seed(seed)
    return


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)


class NeuralNetwork(object):

    def __init__(self, nn_config, verbosity=1):
        self.nn_config = nn_config
        self.x_ph = None
        self.mask_ph = None
        self.obs_y_ph = None
        self.full_outputs = None
        self.training_op = None
        self.loss = None
        self.saver = None
        self.saved_epochs = None
        self.losses = None
        self.verbose = verbosity

    def build(self):

        reset_graph()

        n_steps = self.nn_config['lookback']

        self.x_ph = tf.compat.v1.placeholder(tf.float32, [None, n_steps, self.nn_config['input_features']],
                                             name='x_ph')  # self.nn_config['batch_size']
        self.obs_y_ph = tf.compat.v1.placeholder(tf.float32, [None, 1], name='obs_y_ph')
        self.mask_ph = tf.compat.v1.placeholder(tf.float32, [None, 1], name='mask_ph')  # self.nn_config['batch_size']

        cell = LSTMCell(units=self.nn_config['lstm_units'], activation=self.nn_config['lstm_activation'])

        if self.nn_config['method'] == 'dynamic_rnn':
            rnn_outputs1, states = dynamic_rnn(cell, self.x_ph, dtype=tf.float32)
            rnn_outputs = tf.reshape(rnn_outputs1[:, -1, :], [-1, self.nn_config['lstm_units']])

        elif self.nn_config['method'] == 'keras_lstm_layer':
            rnn_outputs = LSTM(self.nn_config['lstm_units'],
                               activation=self.nn_config['lstm_activation'],
                               input_shape=(n_steps, self.nn_config['input_features']))(self.x_ph)

        else:
            rnn_layer = RNN(cell)
            rnn_outputs = rnn_layer(self.x_ph)  # [batch_size, neurons]
            if self.verbose > 0:
                print(rnn_outputs.shape, 'before reshaping', K.eval(tf.rank(rnn_outputs)))
            rnn_outputs = tf.reshape(rnn_outputs[:, :], [-1, self.nn_config['lstm_units']])
            if self.verbose > 0:
                print(rnn_outputs.shape, 'after reshaping', K.eval(tf.rank(rnn_outputs)))

        if self.nn_config['batch_norm']:
            rnn_outputs3 = BatchNormalization()(rnn_outputs)
            rnn_outputs = Activation('relu')(rnn_outputs3)

        if self.nn_config['dropout'] is not None:
            rnn_outputs = Dropout(self.nn_config['dropout'])(rnn_outputs)

        if self.verbose > 0:
            print(rnn_outputs.shape, "dynamic_rnn/RNN outputs shape", rnn_outputs.get_shape())

        leaky_layer = LeakyDense2D(units=self.nn_config['output_features'], activation=tf.nn.elu, leaky_inputs=True,
                                   mask_array=self.mask_ph, verbose=self.verbose)  # tf.nn.elu
        dense_outputs = leaky_layer(rnn_outputs)

        self.nn_config['dense_activation'] = leaky_layer.activation.__name__

        outputs = tf.reshape(dense_outputs, [-1, 1])  # this is because observations will also be flattened
        self.full_outputs = leaky_layer.full_outputs

        if self.verbose > 0:
            print(outputs.shape, 'shape outputs')

        self.loss = tf.reduce_mean(tf.square(outputs - self.obs_y_ph))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.nn_config['lr'])
        self.training_op = optimizer.minimize(self.loss)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.nn_config['n_epochs'])

    def train(self, train_batches, val_batches, monitor):

        train_x = train_batches[0]
        train_y = train_batches[1]
        val_x = val_batches[0]
        val_y = val_batches[1]

        if len(train_x) != len(train_y):
            raise ValueError("No of Training x batches is {} and y batches are {]. They must be equal"
                             .format(len(train_x), len(train_y)))

        if len(val_x) != len(val_y):
            raise ValueError("No of Training x batches is {} and y batches are {]. They must be equal"
                             .format(len(val_x), len(val_y)))

        no_train_batches = len(train_x)
        no_val_batches = len(val_x)

        train_epoch_losses = OrderedDict({key: [] for key in monitor})
        val_epoch_losses = OrderedDict({key: [] for key in monitor})
        funcs = {'mse': [np.min, np.less, np.argmin, 'min'],
                 'r2': [np.max, np.greater, np.argmax, 'max'],
                 'kge': [np.max, np.greater, np.argmax, 'max'],
                 'nse': [np.max, np.greater, np.argmax, 'max']}

        save_path = os.path.join(os.getcwd(), 'check_points')

        st_t, self.nn_config['start_time'] = time.time(), time.asctime()

        m = [m for m in monitor for _ in range(2)]
        header = ['\t' + i+'_'+j for i, j in zip(m, ['train', 'test']*int(len(m)/2))]
        print('Epoch ', *header, sep=' ', end=' ', flush=True)
        print()

        init = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            writer = tf.compat.v1.summary.FileWriter('./graph/leaky_lstm', sess.graph)
            init.run()

            for epoch in range(self.nn_config['n_epochs']):

                train_batch_losses = OrderedDict({key: [] for key in monitor})

                for bat in range(no_train_batches):
                    x_batch, mask_y_batch = train_x[bat], train_y[bat]

                    # original mask_y_batch can contain have >1 second dimensions but we flatten it into 1D
                    mask_y_batch = mask_y_batch.reshape(-1, 1)  # mask_y_batch will be used to slice dense layer outputs

                    y_of_interest = mask_y_batch[np.where(mask_y_batch > 0.0)].reshape(-1, 1)

                    _, mse, y_pred = sess.run([self.training_op, self.loss, self.full_outputs],
                                              feed_dict={self.x_ph: x_batch, self.obs_y_ph: y_of_interest,
                                                         self.mask_ph: mask_y_batch})

                    # because loss was calculated by flattening all output arrays, so we are calculating loss
                    # here like this
                    # cons: we can not find out individual loss for each observation/target array
                    # however, it will be interesting to visualize individual losses of each observation
                    y_pred = y_pred.reshape(-1, 1)
                    y_pred = y_pred[np.where(mask_y_batch > 0.0)]

                    if len(y_pred) > 1:
                        # y_of_interest = train_y_scaler.inverse_transform(y_of_interest.reshape(-1,output_features))
                        # y_pred = train_y_scaler.inverse_transform(y_pred.reshape(-1, output_features))

                        er = FindErrors(y_of_interest.reshape(-1,), y_pred.reshape(-1,))

                        for error in train_batch_losses.keys():
                            er_val = float(getattr(er, str(error))())
                            train_batch_losses[error].append(er_val)

                # evaluate model on validation dataset
                val_batch_losses = OrderedDict({key: [] for key in monitor})

                for bat in range(no_val_batches):
                    val_x_batch, y_batch = val_x[bat], val_y[bat]

                    y_batch = y_batch.reshape(-1, 1)  # mask_y_batch will be used to slice dense layer output

                    mask_y_batch = np.where(y_batch > 0.0)
                    y_of_interest = y_batch[mask_y_batch[0]].reshape(-1, )

                    y_pred = sess.run(self.full_outputs, feed_dict={self.x_ph: val_x_batch, self.mask_ph: y_batch})

                    # flattening prediction array because of same reason as mentioned above.
                    y_pred = y_pred.reshape(-1, 1)
                    y_pred = y_pred[mask_y_batch]

                    # y_of_interest = val_y_scaler.inverse_transform(y_of_interest.reshape(-1,output_features))
                    # y_pred = val_y_scaler.inverse_transform(y_pred.reshape(-1, output_features))

                    er = FindErrors(y_of_interest.reshape(-1,), y_pred.reshape(-1,))
                    for error in val_batch_losses.keys():
                        er_val = float(getattr(er, str(error))())
                        val_batch_losses[error].append(er_val)

                # aggregating mean losses for current epoch
                ps = ' '
                save_fg = False
                to_save = None
                for error in train_epoch_losses.keys():
                    f1, f2 = funcs[error][0], funcs[error][1]
                    ps, _, save_fg = check_min_loss(train_epoch_losses[error], train_batch_losses[error], epoch, f1, f2,
                                                    ps, save_fg, to_save)
                    to_save = 1
                    ps, _, save_fg = check_min_loss(val_epoch_losses[error], val_batch_losses[error], epoch, f1, f2,
                                                    ps, save_fg, to_save)

                if save_fg:
                    self.saver.save(sess, save_path=save_path,  global_step=epoch)

                print(epoch, ps)

                if epoch > (self.nn_config['n_epochs']-2):
                    self.saver.save(sess, save_path=save_path,  global_step=epoch)

        en_t, self.nn_config['end_time'] = time.time(), time.asctime()
        train_time = (en_t - st_t) / 60.0 / 60.0
        self.nn_config['train_duration'] = int(train_time)
        print('totoal time taken {}'.format(train_time))

        saved_epochs = {}
        for error in train_epoch_losses.keys():
            k = funcs[error][3] + '_train_' + error + '_epoch'
            f2 = funcs[error][2]
            saved_epochs[k] = int(f2(train_epoch_losses[error]))

        for error in val_epoch_losses.keys():
            k = funcs[error][3] + '_test_' + error + '_epoch'
            f2 = funcs[error][2]
            saved_epochs[k] = int(f2(val_epoch_losses[error]))

        self.saved_epochs = saved_epochs
        self.losses = {'train_losses': train_epoch_losses,
                       'val_losses': train_epoch_losses}
        return saved_epochs, train_epoch_losses, val_epoch_losses

    def run_check_point(self, check_point, x_batches, y_batches, data_set, scalers):

        n_outs = self.nn_config['output_features']

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            self.saver.restore(sess, check_point)

            # evaluating model on training data
            test_iterations = len(x_batches)
            print(test_iterations, 'iter')

            # data_set.shape[1] + 1
            x_data = np.full((test_iterations * self.nn_config['batch_size'], self.nn_config['input_features']), np.nan)

            test_y_pred = np.full((test_iterations * self.nn_config['batch_size'], n_outs), np.nan)
            test_y_true = np.full((test_iterations * self.nn_config['batch_size'], n_outs), np.nan)
            st = 0
            en = self.nn_config['batch_size']
            for i in range(test_iterations):
                test_x_batch, mask_y_batch = x_batches[i], y_batches[i]
                # y_of_interest = mask_y_batch[np.where(mask_y_batch>0.0)].reshape(-1,1)

                y_pred = sess.run(self.full_outputs,
                                  feed_dict={self.x_ph: test_x_batch, self.mask_ph: mask_y_batch.reshape(-1, 1)})

                y_scaler = scalers[str(list(scalers.keys())[-1])]
                y_pred_un = y_scaler.inverse_transform(y_pred.reshape(-1, n_outs))
                mask_y_batch_un = y_scaler.inverse_transform(mask_y_batch.reshape(-1, n_outs))

                test_y_pred[st:en, :] = y_pred_un.reshape(-1, n_outs)
                test_y_true[st:en, :] = mask_y_batch_un.reshape(-1, n_outs)

                for dat in range(self.nn_config['input_features']):
                    value = test_x_batch[:, -1, dat].reshape(-1, 1)
                    val_scaler = scalers[str(dat) + '_scaler']
                    new_value = val_scaler.inverse_transform(value.reshape(-1, 1))
                    x_data[st:en, dat] = new_value.reshape(-1, )

                st += self.nn_config['batch_size']
                en += self.nn_config['batch_size']

        return x_data, test_y_pred, test_y_true
