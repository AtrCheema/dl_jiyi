
from data_preparation import DATA
from neural_net import Model
from utils import generate_event_based_batches
from utils import nan_to_num, maybe_create_path

import numpy as np
import os
import pandas as pd
from collections import OrderedDict



data_obj = DATA()
all_data = data_obj.df

in_features = ['pcp_mm', 'wat_temp_c', 'sal_psu', 'wind_speed_mps']
out_features = ['blaTEM_coppml']

df = all_data[in_features].copy()
for out in out_features:
    df[out] = all_data[out].copy()
print(df.shape)
df.head()

dataset = nan_to_num(df.values, len(out_features), replace_with=0.0)



_path = maybe_create_path()
verbosity = 1

data_conf = OrderedDict()
lookback = 3
data_conf['in_features'] = in_features
data_conf['out_features'] = out_features
data_conf['lookback'] = lookback

train_args = {'lookback': lookback,
              'in_features': len(in_features),
              'out_features': len(out_features),
              'future_y_val': 1,
               'trim_last_batch': True
              }


BatchSize = 10
intervals = [np.array([i for i in range(10, 60, BatchSize)]),
             np.array([i for i in range(51, 91, BatchSize)]),

             np.array([i for i in range(92, 202, BatchSize)]),

             np.array([i for i in range(256, 406, BatchSize)])]
x, y = generate_event_based_batches(dataset, BatchSize, train_args, intervals, 2)

lstm_units = 100
lr = 1e-5
dropout = 0.2
act_f = 'relu'

nn_conf = OrderedDict()
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

## initiate model model
model = Model(nn_conf, verbose=verbosity)

# build model
model.build_nn()

# # train model
# saved_epochs, losses = model.train_nn(train_generator, train_gen, test_generator, test_gen)
# dd = pd.DataFrame.from_dict(losses)
# dd.to_csv(_path + '/losses.txt')