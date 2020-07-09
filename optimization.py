from collections import OrderedDict
import tensorflow as tf
import numpy as np

from main import Model
from run_model import make_model


def reset_graph():
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()


def objective_func(batch_size, lookback, lr, lstm_units, lstm_act='relu',
                   cnn_act='relu',
                   filters=64):

    n_epochs = 10000

    data_config, nn_config, args, intervals, verbosity = make_model(int(batch_size),
                                                                    int(lookback),
                                                                    n_epochs,
                                                                    lr,
                                                                    lstm_act,
                                                                    cnn_act,
                                                                    lstm_units,
                                                                    filters)

    model = Model(data_config=data_config,
                  nn_config=nn_config,
                  args=args,
                  intervals=intervals,
                  verbosity=verbosity)

    model.build_nn()
    model.train_nn()

    mse = np.min(model.losses['val_losses']['mse'])

    reset_graph()

    return mse
