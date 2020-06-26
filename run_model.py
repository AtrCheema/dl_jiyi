from collections import OrderedDict
from main import Model


def make_model(batch_size, lookback, n_epochs, lr, lstm_act, cnn_act, lstm_units, filters):

    in_features = ['pcp_mm', 'pcp3_mm', 'tide_cm', 'wat_temp_c', 'sal_psu',
                   'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'rel_hum']
    out_features = ['aac_coppml']

    _data_config = OrderedDict()
    batch_size = int(batch_size)
    _data_config['in_features'] = in_features
    _data_config['out_features'] = out_features
    _data_config['lookback'] = int(lookback)
    _data_config['normalize'] = True
    _data_config['freq'] = '30min'
    _data_config['monitor'] = ['mse']  # , 'r2'
    _data_config['batch_making_mode'] = 'sample_based'

    train_args = {'lookback': _data_config['lookback'],
                  'in_features': len(in_features),
                  'out_features': len(out_features),
                  'future_y_val': 1,
                  'start': 0,
                  'end': 1450,
                  'trim_last_batch': True
                  }

    _nn_config = OrderedDict()

    _nn_config['lstm_conf'] = {'lstm_units': lstm_units,
                               'dropout': 0.3,
                               'lstm_activation': lstm_act,  # will be none if batch_norm is True
                               'method': 'keras_lstm_layer',
                               'batch_norm': False}
    _nn_config['1dCNN'] = {'filters': filters,
                           'kernel_size': 2,
                           'activation': cnn_act,
                           'max_pool_size': 2}

    _nn_config['lr'] = lr
    _nn_config['n_epochs'] = n_epochs
    _nn_config['batch_size'] = batch_size
    _nn_config['loss'] = 'mse'  # options are mse/r2/nse/kge/mae, kge not working yet
    _nn_config['clip_norm'] = 1.0  # None or any scaler value
    _verbosity = 1

    total_intervals = {
        0: [i for i in range(0, 152, batch_size)],
        1: [i for i in range(140, 390, batch_size)],
        2: [i for i in range(380, 630, batch_size)],
        3: [i for i in range(625, 825, batch_size)],
        4: [i for i in range(820, 1110, batch_size)],
        5: [i for i in range(1110, 1447, batch_size)]
    }

    _intervals = {'train_intervals': total_intervals,
                  'test_intervals': total_intervals,
                  'all_intervals': total_intervals}

    _args = {'train_args': train_args.copy(),
             'test_args': train_args.copy(),
             'all_args': train_args.copy()}

    return _data_config, _nn_config, _args, _intervals, _verbosity


data_config, nn_config, args, intervals, verbosity = make_model(batch_size=4,
                                                                lookback=8,
                                                                n_epochs=15,
                                                                lr=1e-6,
                                                                lstm_act='relu',
                                                                cnn_act='relu',
                                                                lstm_units=128,
                                                                filters=64)

model = Model(data_config=data_config,
              nn_config=nn_config,
              args=args,
              intervals=intervals,
              verbosity=verbosity)

model.build_nn()
saved_epochs, losses = model.train_nn()
errors, neg_predictions = model.predict()

# # to load and run checkpoints comment above two lines and uncomment following code
# path = "D:\\dl_jiyi\\models\\20200605_0954"
# model = Model.from_config(path)
# model.build_nn()
# model.predict(mode='all')
