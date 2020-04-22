
from data_preparation import DATA
from neural_net import NeuralNetwork as Model
from utils import generate_event_based_batches
from utils import nan_to_num, maybe_create_path
from utils import normalize_data
# from post_processing import post_process

import numpy as np
import os
import pandas as pd
from collections import OrderedDict

np.printoptions(precision=5)

data_obj = DATA(freq='30min')
all_data = data_obj.df

in_features = ['pcp_mm', 'wat_temp_c', 'sal_psu', 'wind_speed_mps']
out_features = ['blaTEM_coppml']

df = all_data[in_features].copy()
for out in out_features:
    df[out] = all_data[out].copy()
print(df.shape)
df.head()

dataset = nan_to_num(df.values, len(out_features), replace_with=0.0)

data_conf = OrderedDict()
lookback = 8
BatchSize = 24
data_conf['in_features'] = in_features
data_conf['out_features'] = out_features
data_conf['lookback'] = lookback
data_conf['normalize'] = True


if data_conf['normalize']:
    dataset, scalers = normalize_data(dataset)

_path = maybe_create_path()
verbosity = 1

train_args = {'lookback': lookback,
              'in_features': len(in_features),
              'out_features': len(out_features),
              'future_y_val': 1,
              'trim_last_batch': True
              }

train_intervals = [np.array([i for i in range(0, 50, BatchSize)]),
                   np.array([i for i in range(51, 91, BatchSize)]),

                   np.array([i for i in range(82, 197, BatchSize)]),

                   np.array([i for i in range(245, 406, BatchSize)]),
                   np.array([i for i in range(440, 524, BatchSize)]),
                   np.array([i for i in range(540, 610, BatchSize)])]
train_x, train_y = generate_event_based_batches(dataset, BatchSize, train_args, train_intervals, 2)

test_intervals = [np.array([i for i in range(737, 780, BatchSize)]),
                  np.array([i for i in range(900, 1177, BatchSize)])
                  ]
test_x, test_y = generate_event_based_batches(dataset, BatchSize, train_args, test_intervals, 2)

test_intervals = [np.array([i for i in range(0, 780, BatchSize)])]
full_x, full_y = generate_event_based_batches(dataset, BatchSize, train_args, test_intervals, 0, raise_error=False)

nn_conf = OrderedDict()
lstm_units = 100
lr = 1e-5
dropout = 0.2
act_f = 'relu'
nn_conf['lstm_units'] = int(lstm_units)
nn_conf['lr'] = lr
nn_conf['method'] = 'keras_lstm_layer'
nn_conf['dropout'] = dropout
nn_conf['batch_norm'] = False
nn_conf['lstm_activation'] = None if nn_conf['batch_norm'] else act_f
nn_conf['n_epochs'] = 10

nn_conf['lookback'] = lookback
nn_conf['input_features'] = len(in_features)
nn_conf['output_features'] = len(out_features)
nn_conf['batch_size'] = BatchSize
nn_conf['monitor'] = ['mse', 'nse', 'kge', 'r2']

# # initiate model model
model = Model(nn_conf, verbose=verbosity)

# build model
model.build_nn()

# # train model
saved_epochs, train_losses, val_losses = model.train(train_batches=[train_x, train_y],
                                                     val_batches=[test_x, test_y],
                                                     monitor=nn_conf['monitor'])
# post_process(data_conf=data_conf,
#              x_batches, y_batches, test_dataset,
#                  model, saved_epochs, _path, all_scalers,  full_args,
#                  losses, verbose=1)
