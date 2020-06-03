
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
from utils import generate_sample_based_batches
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
    """ Just to store attributes of Model class"""
    scalers = {key: None for key in ['train', 'test', 'all']}  # AttributeNotSetYet('build_nn')

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
            dataset, self.scalers['all'] = normalize_data(dataset, df.columns, 1)

        return dataset  # , scalers

    def get_batches_new(self, mode):

        print('\n', '*' * 14)
        print("creating data for {} mode".format(mode))
        in_features = self.data_config['in_features']
        out_features = self.data_config['out_features']

        data_obj = DATA(freq=self.data_config['freq'])
        all_data = data_obj.get_df_from_rf('opt_set.mat')  # INPUT

        # copying required data
        df = all_data[in_features].copy()
        for out in out_features:
            df[out] = all_data[out].copy()
        if self.verbosity > 0:
            print('shape of whole dataset', df.shape)

        # assuming that pandas will add the 'datetime' column as last column. This columns will only be used to keep
        # track of indices of train and test data.
        df['datetime'] = list(map(int, np.array(df.index.strftime('%Y%m%d%H%M'))))

        index = all_data[mode + '_index']
        ttk = index.dropna()

        self.args[mode + '_args']['no_of_samples'] = len(ttk)

        ttk_idx = list(map(int, np.array(ttk.index.strftime('%Y%m%d%H%M'))))  # list

        df['to_keep'] = 0
        df['to_keep'][ttk.index] = ttk_idx

        dataset = nan_to_num(df.values, len(out_features)+2, replace_with=0.0)

        if self.data_config['normalize']:
            dataset, self.scalers[mode] = normalize_data(dataset, df.columns, 2)

        self.batches[mode + '_x'],\
            self.batches[mode + '_y'], \
            self.nn_config[mode + '_no_of_batches'], \
            self.batches[mode + '_index'],\
            self.batches[mode + '_tk_index'] = generate_sample_based_batches(self.args[mode + '_args'],
                                                                             self.nn_config['batch_size'],
                                                                             dataset)
        print('*' * 14, '\n')
        return

    def get_batches(self, dataset, mode):

        skip_batch_with_no_labels = True
        raise_errors = True
        if self.data_config['batch_making_mode'] == 'sample_based':
            st = self.args['all_args']['start']
            en = self.args['all_args']['end']
            self.intervals = {'all_intervals': [[i for i in range(st, en, self.nn_config['batch_size'])]]}
            raise_errors = False
            skip_batch_with_no_labels = False

        self.batches[mode + '_x'],\
            self.batches[mode + '_y'],\
            self.nn_config[mode + '_no_of_batches'],\
            self.batches[mode + '_index'],\
            self.data_config['no_of_' + mode + '_samples'] = generate_event_based_batches(
                dataset,
                self.nn_config['batch_size'],
                self.args[mode + '_args'],
                self.intervals[mode + '_intervals'],
                self.verbosity,
                raise_error=raise_errors,
                skip_batch_with_no_labels=skip_batch_with_no_labels)

        return

    def build_nn(self):

        dataset = self.pre_process_data()

        if "batch_making_mode" in self.data_config:
            if self.data_config["batch_making_mode"] == "sample_based":

                for mode in ['train', 'test']:
                    self.get_batches_new(mode)

            else:
                for mode in ['train', 'test']:
                    self.get_batches(dataset=dataset, mode=mode)
        else:
            for mode in ['train', 'test']:
                self.get_batches(dataset=dataset, mode=mode)

        self.get_batches(dataset, mode='all')

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

    def predict(self, mode=None, epochs_to_eval=None):
        """
        :param mode: list or str, if list then all members must be str default is ['train', 'test', 'all']
        :param epochs_to_eval: int or list of integers
        :return: errors, dictionary of errors from each mode
                 neg_predictions, dictionary of negative prediction from each mode
        """

        mode = _get_mode(mode)

        if epochs_to_eval is None:
            epochs_to_evaluate = self.data_config['saved_unique_cp']
        else:
            if isinstance(epochs_to_eval, int):
                epochs_to_evaluate = [epochs_to_eval]
            elif isinstance(epochs_to_eval, list):
                epochs_to_evaluate = epochs_to_eval
            else:
                raise TypeError

        errors = {}
        neg_predictions = {}
        for m in mode:
            if self.verbosity > 0:
                stars = "************************************************"
                print(stars, "\nPrediction using {} data\n".format(m), stars)
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
        # config["saved_epochs"] = self.saved_epochs
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

        if "batch_making_mode" in self.data_config:
            pass
        else:
            validate_dictionary(self.intervals, INTERVALS_KEYS, 'intervals')
            self.data_config['batch_making_mode'] = 'event_based'

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

    def print_samples(self, mode='train', data_type='y'):

        batch_name = mode + '_' + data_type
        total_samples = 0

        for i in range(self.batches[batch_name].shape[0]):
            batch = self.batches[batch_name][i, :]
            vals = batch[:, 0]
            nzs = vals[np.where(vals > 0.0)]
            print('Batch: ', i, nzs)
            total_samples += len(nzs)

        print('Total samples are: {}'.format(total_samples))

        return


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
