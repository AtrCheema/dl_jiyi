
from collections import OrderedDict

from main import Model

in_features = ['pcp_mm', 'wat_temp_c', 'sal_psu', 'wind_speed_mps']
out_features = ['blaTEM_coppml']

data_config = OrderedDict()
lookback = 8
BatchSize = 32
data_config['in_features'] = in_features
data_config['out_features'] = out_features
data_config['normalize'] = True
data_config['freq'] = '30min'

train_args = {'lookback': lookback,
              'in_features': len(in_features),
              'out_features': len(out_features),
              'future_y_val': 1,
              'trim_last_batch': True
              }

train_intervals = [
    [i for i in range(63, 100, BatchSize)],
    [i for i in range(187, 230, BatchSize)],
    [i for i in range(230, 355, BatchSize)],
    [i for i in range(450, 600, BatchSize)],

    [i for i in range(639, 715, BatchSize)],
    [i for i in range(740, 800, BatchSize)],
    # [i for i in range(870, 910, BatchSize)],
    [i for i in range(910, 1015, BatchSize)]
]

test_intervals = [
    [i for i in range(1150, 1422, BatchSize)]
]

all_intervals = [
    [i for i in range(0, 1422, BatchSize)]
                 ]


nn_config = OrderedDict()
lstm_units = 100
lr = 1e-5
dropout = 0.2
act_f = 'relu'
nn_config['lstm_units'] = int(lstm_units)
nn_config['lr'] = lr
nn_config['method'] = 'keras_lstm_layer'
nn_config['dropout'] = dropout
nn_config['batch_norm'] = False
nn_config['lstm_activation'] = None if nn_config['batch_norm'] else act_f
nn_config['n_epochs'] = 10

nn_config['lookback'] = lookback
nn_config['input_features'] = len(in_features)
nn_config['output_features'] = len(out_features)
nn_config['batch_size'] = BatchSize
data_config['monitor'] = ['mse', 'nse', 'kge', 'r2']

verbosity = 1

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
# model.train_nn()
# model.post_process()

# to load and run checkpoints comment above two lines and uncomment following code
# path = d = "D:\\dl_jiyi\\models\\20200422_2009"
# model = Model.from_config(path)
# model.build_nn()
# model.post_process(from_config=True)
