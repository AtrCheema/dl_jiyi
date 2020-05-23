
from data_preparation import DATA
from neural_net import NeuralNetwork as nn
from utils import generate_event_based_batches
from utils import nan_to_num, maybe_create_path
from utils import normalize_data
from utils import validate_dictionary
from utils import save_config_file
from utils import plot_loss
from utils import copy_check_points
from utils import AttributeNotSetYet
from post_processing import make_predictions

import os
import numpy as np
from collections import OrderedDict
import json
import pandas as pd

np.printoptions(precision=5)

DATA_CONF_KEYS = ['in_features', 'out_features', 'normalize', 'freq', 'monitor']

NN_CONF_KEYS = ['lstm_units', 'lr', 'method', 'dropout', 'batch_norm', 'lstm_activation',
                'n_epochs', 'lookback', 'input_features', 'output_features', 'batch_size', 'loss']

INTERVALS_KEYS = ['train_intervals', 'test_intervals', 'all_intervals']

ARGS_KEYS = ['train_args', 'test_args', 'all_args']


class ModelAttr(object):

    scalers = AttributeNotSetYet('build_nn')

    def __init__(self):
        pass


class Model(nn, ModelAttr):

    def __init__(self, data_config,
                 nn_config,
                 args,
                 intervals,
                 path=None,   # if specified, and if it exists, will not be created then
                 verbosity=1):

        self.data_config = data_config
        self.nn_config = nn_config
        self.args = args
        self.intervals = intervals  # dictionary
        self.verbosity = verbosity
        self._validate_input()
        self.batches = {}
        self.path = maybe_create_path(path=path)
        self._from_config = False if path is None else True

        super(Model, self).__init__(nn_config=nn_config,
                                    data_config=data_config,
                                    path=self.path,
                                    verbosity=verbosity)

    def pre_process_data(self):
        in_features = self.data_config['in_features']
        out_features = self.data_config['out_features']

        data_obj = DATA(freq=self.data_config['freq'])
        all_data = data_obj.get_df()

        # copying required data
        df = all_data[in_features].copy()
        for out in out_features:
            df[out] = all_data[out].copy()
        if self.verbosity > 0:
            print('shape of whole dataset', df.shape)

        # assuming that pandas will add the 'datetime' column as last column. This columns will only be used to keep
        # track of indices of train and test data.
        df['datetime'] = list(map(int, np.array(df.index.strftime('%Y%m%d%H%M'))))

        # columns containing target data (may) have nan values because missing values are represented by nans
        # so convert those nans 0s. This is with a big assumption that the actual target data does not contain 0s.
        # they are converted to zeros because in LSTM and at other places as well we will select the data based on mask
        # such as values>0.0 and if target data has zeros, we can not do this.
        dataset = nan_to_num(df.values, len(out_features)+1, replace_with=0.0)

        if self.data_config['normalize']:
            dataset, self.scalers = normalize_data(dataset)

        return dataset  # , scalers

    def get_batches(self, dataset):

        train_x, train_y, train_no_of_batches, train_indexes,\
            self.data_config['no_of_train_samples'] = generate_event_based_batches(dataset,
                                                                                   self.nn_config['batch_size'],
                                                                                   self.args['train_args'],
                                                                                   self.intervals['train_intervals'],
                                                                                   self.verbosity,
                                                                                   skip_batch_with_no_labels=True)

        test_x, test_y, test_no_of_batches, test_index,\
            self.data_config['no_of_test_samples'] = generate_event_based_batches(dataset,
                                                                                  self.nn_config['batch_size'],
                                                                                  self.args['train_args'],
                                                                                  self.intervals['test_intervals'],
                                                                                  self.verbosity,
                                                                                  skip_batch_with_no_labels=True)

        all_x, all_y, all_no_of_batches, all_indexes,\
            self.data_config['no_of_all_samples'] = generate_event_based_batches(dataset,
                                                                                 self.nn_config['batch_size'],
                                                                                 self.args['train_args'],
                                                                                 self.intervals['all_intervals'],
                                                                                 self.verbosity-1,
                                                                                 raise_error=False)

        self.nn_config['train_no_of_batches'] = train_no_of_batches
        self.nn_config['test_no_of_batches'] = test_no_of_batches
        self.nn_config['all_no_of_batches'] = all_no_of_batches

        self.batches['train_x'] = train_x
        self.batches['train_y'] = train_y
        self.batches['test_x'] = test_x
        self.batches['test_y'] = test_y
        self.batches['all_x'] = all_x
        self.batches['all_y'] = all_y
        self.batches['train_index'] = train_indexes.astype(np.int64)
        self.batches['test_index'] = test_index.astype(np.int64)
        self.batches['all_index'] = all_indexes.astype(np.int64)
        return

    def build_nn(self):

        dataset = self.pre_process_data()

        self.get_batches(dataset)

        # build neural network
        self.build()

    def train_nn(self):

        # # train model
        self.train(train_batches=[self.batches['train_x'], self.batches['train_y']],
                   val_batches=[self.batches['test_x'], self.batches['test_y']],
                   monitor=self.data_config['monitor'])

        self.handle_losses()

        saved_unique_cp = copy_check_points(self.saved_epochs, os.path.join(self.path, 'check_points'))
        self.data_config['saved_unique_cp'] = saved_unique_cp

        self.save_config()
        self.remove_redundant_epochs()

        return self.saved_epochs, self.losses

    def predict(self, mode=None):
        """
        :param mode: list or str, if list then all members must be str default is ['train', 'test', 'all']
        :return: errors, dictionary of errors from each mode
                 neg_predictions, dictionary of negative prediction from each mode
        """

        mode = _get_mode(mode)

        epochs_to_evaluate = self.data_config['saved_unique_cp']

        errors = {}
        neg_predictions = {}
        for m in mode:
            _errors, _neg_predictions = make_predictions(x_batches=self.batches[m + '_x'],  # like `train_x` or `val_x`
                                                         y_batches=self.batches[m + '_y'],
                                                         model=self,
                                                         epochs_to_evaluate=epochs_to_evaluate,
                                                         runtype=m,
                                                         save_results=True)

            errors[m + '_errors'] = _errors
            neg_predictions[m + '_neg_predictions'] = _neg_predictions

        self.save_config(errors=errors, neg_predictions=neg_predictions)

        return errors, neg_predictions

    def save_config(self, errors=None, neg_predictions=None):

        config = OrderedDict()
        config['comment'] = 'use point source pollutant data along with best model from grid search'
        config['nn_config'] = self.nn_config
        config['data_config'] = self.data_config
        config['test_errors'] = errors
        config['test_sample_idx'] = 'test_idx'
        config['start_time'] = self.nn_config['start_time'] if 'start_time' in self.nn_config else " "
        config['end_time'] = self.nn_config['end_time'] if 'end_time' in self.nn_config else " "
        config["saved_epochs"] = self.saved_epochs
        config['intervals'] = self.intervals
        config['args'] = self.args
        config['train_time'] = self.nn_config['train_duration'] if 'train_duration' in self.nn_config else " "
        config['final_comment'] = """ """
        config['negative_predictions'] = neg_predictions

        save_config_file(config, self.path, from_config=self._from_config)
        return config

    def handle_losses(self):

        if self.losses is not None:
            for loss_type, loss in self.losses.items():
                pd.DataFrame.from_dict(loss).to_csv(self.path + '/' + loss_type + '.txt')

            # plot losses
            for er in self.data_config['monitor']:
                plot_loss(self.losses['train_losses'][er], self.losses['val_losses'][er], er, self.path)
        return

    @classmethod
    def from_config(cls, _path):

        config_file = os.path.join(_path, 'config.json')
        with open(config_file, 'r') as fp:
            data = json.load(fp)

        intervals = data['intervals']

        args = data['args']

        nn_config = data['nn_config']

        data_config = data['data_config']

        return cls(data_config=data_config,
                   nn_config=nn_config,
                   args=args,
                   intervals=intervals,
                   path=_path,
                   verbosity=1)

    def _validate_input(self):

        validate_dictionary(self.data_config, DATA_CONF_KEYS, 'data_config')

        validate_dictionary(self.nn_config, NN_CONF_KEYS, 'nn_config')

        validate_dictionary(self.intervals, INTERVALS_KEYS, 'intervals')

        validate_dictionary(self.args, ARGS_KEYS, 'args')

    def remove_redundant_epochs(self):
        all_epochs = find_saved_epochs(os.path.join(self.path, 'check_points'))
        all_ep = [int(i) for i in all_epochs]
        to_keep = list(self.saved_epochs.values())
        to_del = []
        for epoch in all_ep:
            if epoch not in to_keep:
                to_del.append(epoch)

        for epoch in to_del:
            fpath = os.path.join(self.path, 'check_points')
            files = ['.index', '.meta', '.data-00000-of-00001']
            for f in files:
                fname = os.path.join(fpath, 'checkpoints-' + str(epoch) + f)
                if os.path.exists(fname):
                    os.remove(fname)


def _get_mode(mode):

    def_mode = ['train', 'test', 'all']
    if mode is None:
        mode = def_mode
    else:
        if not isinstance(mode, list):
            if not isinstance(mode, str):
                raise TypeError("mode must be string")
            else:
                mode = [mode]
        else:
            for m in mode:
                if m not in def_mode:
                    raise ValueError("{} not allowed".format(m))
    return mode


def find_saved_epochs(_path):
    idx_files = [f for f in os.listdir(_path) if f.endswith('.index')]
    saved_epochs = [f.split('-')[1].split('.')[0] for f in idx_files]
    return list(np.unique(saved_epochs))
