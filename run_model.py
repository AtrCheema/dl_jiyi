
from collections import OrderedDict
import random
import numpy as np
from main import Model

in_features = ['pcp_mm', 'pcp3_mm', 'tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa',  'rel_hum']
out_features = ['blaTEM_coppml']

data_config = OrderedDict()
BatchSize = 4
data_config['in_features'] = in_features
data_config['out_features'] = out_features
data_config['lookback'] = 8
data_config['normalize'] = True
data_config['freq'] = '30min'
data_config['monitor'] = ['mse']  # , 'r2'
data_config['batch_making_mode'] = 'sample_based'

train_args = {'lookback': data_config['lookback'],
              'in_features': len(in_features),
              'out_features': len(out_features),
              'future_y_val': 1,
              'start': 0,
              'end': 1450,
              'trim_last_batch': True
              }


nn_config = OrderedDict()

# nn_config['lstm_conf'] = {'lstm_units': 128,
#                           'dropout': 0.3,
#                           'lstm_activation': 'relu',  # will be none if batch_norm is True
#                           'method': 'keras_lstm_layer',
#                           'batch_norm': False}
nn_config['1dCNN'] = {'filters': 64,
                      'kernel_size': 2,
                      'activation': 'relu',
                      'max_pool_size': 2}

nn_config['lr'] = 1e-6
nn_config['n_epochs'] = 15
nn_config['batch_size'] = BatchSize
nn_config['loss'] = 'mse'   # options are mse/r2/nse/kge/mae, kge not working yet
nn_config['clip_norm'] = 1.0  # None or any scaler value
verbosity = 1


total_intervals = {
    0: [i for i in range(0, 152, BatchSize)],
    1: [i for i in range(140, 390, BatchSize)],
    2: [i for i in range(380, 630, BatchSize)],
    3: [i for i in range(625, 825, BatchSize)],
    4: [i for i in range(820, 1110, BatchSize)],
    5: [i for i in range(1110, 1447, BatchSize)]
}

intervals = {'train_intervals': total_intervals,
             'test_intervals': total_intervals,
             'all_intervals': total_intervals}

args = {'train_args': train_args.copy(),
        'test_args': train_args.copy(),
        'all_args': train_args.copy()}

model = Model(data_config=data_config,
              nn_config=nn_config,
              args=args,
              intervals=intervals,
              verbosity=verbosity)

model.build_nn()
saved_epochs, losses = model.train_nn()
errors, neg_predictions = model.predict()

# # to load and run checkpoints comment above two lines and uncomment following code
# path = "D:\\dl_jiyi\\models\\20200604_0023"
# model = Model.from_config(path)
# model.build_nn()
# model.predict()
