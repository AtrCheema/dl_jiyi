
from collections import OrderedDict
import tensorflow as tf
import random

from main import Model


def reset_graph():
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()


def objective_func(pot_tr_intervals, BatchSize=24, lookback=12, lr=1e-6, lstm_units=128, act_f='relu'):

    in_features = ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum']

    out_features = ['blaTEM_coppml']

    nn_config = OrderedDict()
    # lstm_units = 100
    # lr = 1e-5
    dropout = 0.2
    # act_f = 'relu'
    nn_config['lstm_units'] = int(lstm_units)
    nn_config['lr'] = lr
    nn_config['method'] = 'keras_lstm_layer'
    nn_config['dropout'] = dropout
    nn_config['batch_norm'] = False
    nn_config['lstm_activation'] = None if nn_config['batch_norm'] else act_f
    nn_config['n_epochs'] = 15000

    nn_config['lookback'] = lookback
    nn_config['input_features'] = len(in_features)
    nn_config['output_features'] = len(out_features)
    nn_config['batch_size'] = BatchSize
    nn_config['loss'] = 'mse'  # options are mse/r2/nse/kge, kge not working yet

    data_config = OrderedDict()
    # lookback = 8
    # BatchSize = 32
    data_config['in_features'] = in_features
    data_config['out_features'] = out_features
    data_config['normalize'] = True
    data_config['freq'] = '30min'
    data_config['monitor'] = ['mse']  # , 'r2'

    verbosity = 1



    train_args = {'lookback': lookback,
                  'in_features': len(in_features),
                  'out_features': len(out_features),
                  'future_y_val': 1,
                  'trim_last_batch': True
                  }

    total_intervals = {
        0: [i for i in range(0, 156, BatchSize)],
        1: [i for i in range(136, 407, BatchSize)],
        2: [i for i in range(412, 546, BatchSize)],
        3: [i for i in range(509, 580, BatchSize)],
        4: [i for i in range(533, 660, BatchSize)],
        5: [i for i in range(633, 730, BatchSize)],
        6: [i for i in range(730, 831, BatchSize)],
        7: [i for i in range(821, 971, BatchSize)],
        8: [i for i in range(941, 1071, BatchSize)],
        9: [i for i in range(1125, 1200, BatchSize)],
        10: [i for i in range(1172, 1210, BatchSize)],
        11: [i for i in range(1196, 1240, BatchSize)],
        12: [i for i in range(1220, 1317, BatchSize)],
        13: [i for i in range(1292, 1335, BatchSize)],
        14: [i for i in range(1316, 1447, BatchSize)]
    }


    train_intervals = []
    test_intervals = []

    for key in total_intervals.keys():
        if key in pot_tr_intervals:
            train_intervals.append(total_intervals[key])
            print('train: ', key)
        else:
            test_intervals.append(total_intervals[key])
            print('test: ', key)

    all_intervals = list(total_intervals.values())

    intervals = {'train_intervals': train_intervals,
                 'test_intervals': test_intervals,
                 'all_intervals': all_intervals}

    args = {'train_args': train_args.copy(),
            'test_args': train_args.copy(),
            'all_args': train_args.copy()}

    model = Model(data_config=data_config,
                  nn_config=nn_config,
                  args=args,
                  intervals=intervals,
                  verbosity=verbosity)

    model.build_nn()
    model.train_nn()
    errors, neg_predictions = model.predict()

    mse = model.losses['val_losses']['mse'][-1]
    last_epoch = model.nn_config['n_epochs'] - 1
    k = str(last_epoch) + '_blaTEM_coppml'
    mse_from_pred = errors['test_errors'][k]['mse']

    reset_graph()

    return mse, mse_from_pred
