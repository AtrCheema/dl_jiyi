
from collections import OrderedDict
import tensorflow as tf


from main import Model


def reset_graph():
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()


def objective_func(BatchSize, lookback, lr, lstm_units, act_f='relu'):

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
    nn_config['n_epochs'] = 15

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

    train_intervals = [
        # [i for i in range(0, 147, BatchSize)],
        # [i for i in range(149, 393, BatchSize)],
        # [i for i in range(394, 638, BatchSize)],
        # [i for i in range(639, 834, BatchSize)],
        # [i for i in range(839, 1100, BatchSize)]
        [i for i in range(0, 138, BatchSize)],
        [i for i in range(204, 362, BatchSize)],
        [i for i in range(357, 431, BatchSize)],
        [i for i in range(567, 708, BatchSize)],
        [i for i in range(705, 807, BatchSize)],
        [i for i in range(871, 1055, BatchSize)],
        [i for i in range(1045, 1115, BatchSize)],
        [i for i in range(1239, 1446, BatchSize)]

    ]

    test_intervals = [
        # [i for i in range(980, 1398, BatchSize)]
        [i for i in range(136, 208, BatchSize)],
        [i for i in range(430, 568, BatchSize)],
        [i for i in range(804, 885, BatchSize)],
        [i for i in range(1119, 1243, BatchSize)]

    ]

    all_intervals = [
        # [i for i in range(0, 1398, BatchSize)]
        [i for i in range(0, 1446, BatchSize)]

    ]

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
