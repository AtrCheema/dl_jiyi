
import os
import numpy as np
import random
import datetime
from copy import deepcopy
from collections import OrderedDict
from shutil import copyfile
from TSErrors import FindErrors
import json
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.dates as mdates


def random_array(_length, lower=0.0, upper=0.5):
    """This creates a random array of length `length` and the floats vary between `lower` and `upper`."""
    rand_array = np.zeros(_length)
    for i in range(_length):
        rand_array[i] = random.uniform(lower, upper)
    return rand_array


colors = {'pcp12': np.array([0.07233712, 0.470282, 0.24355425]),
          'pcp6': np.array([0.21407651, 0.23047467, 0.69924557]),
          'pcp_mm': np.array([0.21407651, 0.23047467, 0.69924557]),
          'sur_q_cha': np.array([0.76985966, 0.33074703, 0.22861025]),
          'pred': np.array([0.81831849, 0.17526342, 0.4766505]),
          'Predicted Bacteria': np.array([0.94577241, 0.08725546, 0.11906984]),
          'pcp3': np.array([0.38079258, 0.17830983, 0.78165943]),
          'pcp1': np.array([0.74503799, 0.72589453, 0.91296436]),
          'tide_cm': np.array([0.39510605, 0.11174541, 0.90067133]),
          'air_temp_c': np.array([0.9624212,  0.28477194, 0.4760216]),
          'wat_temp_c': np.array([0.51618149, 0.16053867, 0.45268923]),
          'wind_speed_mps': np.array([0.76985966, 0.33074703, 0.22861025]),
          'sal_psu': np.array([0.76985966, 0.33074703, 0.22861025]),
          'Total Discharge': np.array([0.38079258, 0.17830983, 0.78165943]),
          'wind_dir_deg': np.array([0.38079258, 0.17830983, 0.78165943]),
          'ecoli': np.array([0.38079258, 0.17830983, 0.78165943]),
          'air_p_hpa': np.array([0.13778617, 0.06228198, 0.33547859]),
          'mslp_hpa': np.array([0.13778617, 0.06228198, 0.33547859]),
          'inti1': np.array([0.96707953, 0.46268314, 0.45772886]),
          'Training NSE': np.array([0.13778617, 0.06228198, 0.33547859]),
          'Validation NSE': np.array([0.96707953, 0.46268314, 0.45772886]),
          'aac_coppml': np.array([0.17221373, 0.53023578, 0.96788307]),
          'blaTEM_coppml': np.array([0.66413778, 0.35891819, 0.69004812]),
          'Total_otus': np.array([0.92875036, 0.09364162, 0.33348078]),
          'Total_args': np.array([0.93950089, 0.64582256, 0.16928645]),
          'tetx_coppml': np.array([0.06802773, 0.46382623, 0.49007703]),
          'otu_94': np.array([0.13684922, 0.98802401, 0.34518303]),
          'otu_5575': np.array([0.54829269, 0.15069842, 0.06147751]),
          'sul1_coppml': np.array([0.26900851, 0.96337978, 0.94641933]),
          'otu_273': np.array([0.95896577, 0.58394066, 0.04189788]),
          '16s': np.array([0.17877267, 0.78893675, 0.92613355]),
          'true': np.array([0.94577241, 0.08725546, 0.11906984]),
          'rel_hum':  np.array([0.56884807, 0.27000573, 0.03299844])
          }


labels = {
    "pcp1": "1 hour commulative rain",
    "pcp_mm": "Precipitation",
    "pcp3": "3 hour commulative rain",
    "pcp6": "6 hour commulative rain",
    "pcp12": "12 hour commulative rain",
    "ecoli": "E-Coli",
    "16s": "16s",
    "inti1": "inti1 ",
    "tide_cm": "Tide",
    "wat_temp_c": "Water Temperature",
    "sal_psu": "Salinity",
    "wind_speed_mps": "Wind speed",
    "wind_dir_deg": "Wind direction ",
    "air_temp_c": "Atmospheric Temperature",
    "air_p_hpa": "Atmospheric Pressure",
    "mslp_hpa": "Mean Sea Level Pressure",
    "rel_hum": "Relative Humidity",

    "aac_coppml": "aac(6')-lb-cr",
    "blaTEM_coppml": "blaTEM",
    "sul1_coppml": "sul1",
    "tetx_coppml": "tetX",
    "Total_args": "Total ARGs",
    "otu_94": "OTU 94",
    "otu_273": "OTU 273",
    "otu_5575": "OTU 5575",
    "Total_OTUs": "Total OTUs"
}

y_labels = {
    "pcp1": "mm",
    "pcp3": "mm",
    "pcp6": "mm",
    "pcp12": "mm",
    "pcp_mm": "mm",
    "ecoli": "MPN per 100 mL",
    "16s": "Copies per mL",
    "inti1": "Copies per mL",
    "tide_cm": "cm",
    "wat_temp_c": r"$^\circ$C",
    "sal_psu": "PSU",
    "wind_speed_mps": "$ms{-1}$",
    "wind_dir_deg": r"$^\circ$",
    "air_temp_c": r"$^\circ$C",
    "air_p_hpa": "hPa",
    "mslp_hpa": "hPa",
    "rel_hum": "%",

    "aac_coppml": "Copies per mL",
    "blaTEM_coppml": "copies per mL",
    "sul1_coppml": "copies per mL",
    "tetx_coppml": "copies per mL",
    "Total_args": "Copies per mL",
    "otu_94": "reads",
    "otu_273": "reads",
    "otu_5575": "reads",
    "Total_OTUs": "reads"
}


def process_axis(axis,
                 data,
                 style='.',
                 c=None,
                 ylim=None,    # limit for y axis
                 x_label="Time",
                 xl_fs=14,
                 y_label=None,
                 yl_fs=14,                        # ylabel font size
                 yl_c='k',        # y label color, if 'same', c will be used else black
                 leg_pos="best",
                 label=None,  # legend, none means do not show legend
                 ms=4,  # markersize
                 leg_fs=16,
                 leg_ms=4,  # legend scale
                 leg_cols=1,  # legend columns, default means all legends will be shown in one columns
                 leg_mode=None,
                 leg_frameon=None,  # turn on/off legend box, default is matplotlib's default
                 xtp_ls=12,  # x tick_params labelsize
                 ytp_ls=12,  # x tick_params labelsize
                 xtp_c='k',    # x tick colors if 'same' c will be used else black
                 ytp_c='k',    # y tick colors, if 'same', c will be used else else black
                 log=False,
                 show_xaxis=True,
                 top_spine=True,
                 bottom_spine=True,
                 invert_yaxis=False,
                 verbose=True):

    if c is None:
        if label in colors:
            c = colors[label]
        else:
            c = random_array(3, 0.01, 0.99)
            if verbose:
                print('for ', label, c)

    in_label = label
    if label in labels:
        label = labels[label]
    axis.plot(data, style, markersize=ms, color=c, label=label, linewidth=ms)

    ylc = c
    if yl_c != 'same':
        ylc = 'k'

    if label is not None:
        axis.legend(loc=leg_pos, fontsize=leg_fs, markerscale=leg_ms, ncol=leg_cols, mode=leg_mode, frameon=leg_frameon)

    # if no y_label is provided, it will checked in y_labels dictionary if not there, ' ' will be used.
    if y_label is None:
        if in_label in y_labels:
            y_label = y_labels[in_label]
        else:
            y_label = ' '
    axis.set_ylabel(y_label, fontsize=yl_fs, color=ylc)

    if log:
        axis.set_yscale('log')

    if invert_yaxis:
        axis.set_ylim(axis.get_ylim()[::-1])

    if ylim is not None:
        if not isinstance(ylim, tuple):
            raise TypeError("ylim must be tuple {} provided".format(ylim))
        axis.set_ylim(ylim)

    xtpc = c
    if xtp_c != 'same':
        xtpc = 'k'

    ytpc = c
    if ytp_c != 'same':
        ytpc = 'k'

    axis.tick_params(axis="x", which='major', labelsize=xtp_ls, colors=xtpc)
    axis.tick_params(axis="y", which='major', labelsize=ytp_ls, colors=ytpc)

    axis.get_xaxis().set_visible(show_xaxis)

    if show_xaxis:
        axis.set_xlabel(x_label, fontsize=xl_fs)

    axis.spines['top'].set_visible(top_spine)
    axis.spines['bottom'].set_visible(bottom_spine)

    # loc = mdates.AutoDateLocator(minticks=4, maxticks=6)
    # axis.xaxis.set_major_locator(loc)
    # fmt = mdates.AutoDateFormatter(loc)
    # axis.xaxis.set_major_formatter(fmt)

    return


def set_fig_dim(fig, width, height):
    fig.set_figwidth(width)
    fig.set_figheight(height)


def do_plot(data, cols, st=None, en=None, save_name=None, pre_train=False, sim_ms=4, obs_logy=False, p_ylim=None,
            single_ax_plots=None):

    if st is None:
        st = data.index[0]
    if en is None:
        en = data.index[-1]

    no_of_plots = len(cols)  # data.shape[1]
    if single_ax_plots is not None:
        if not isinstance(single_ax_plots, list):
            raise TypeError
        no_of_plots += 1

        for name in single_ax_plots:
            if name in cols:
                cols.remove(name)
                no_of_plots -= 1

    _fig, axis = plt.subplots(no_of_plots, sharex='all')
    set_fig_dim(_fig, 19, 16)

    idx = 0
    for ax in axis:
        if no_of_plots-1 > idx > 0:  # middle plots
            style = '-'
            invert_yaxis = False
            if 'pcp' in cols[idx]:
                style = '-'
                invert_yaxis = True
            _data = data[cols[idx]][st:en].values
            process_axis(ax, _data, style='o', ms=4, label=cols[idx])
            process_axis(ax, _data, style=style, ms=4, label=cols[idx], show_xaxis=False, bottom_spine=False, leg_fs=14,
                         invert_yaxis=invert_yaxis, verbose=True)

        elif idx == no_of_plots-1:  # last
            if single_ax_plots is not None:
                for col in single_ax_plots:
                    val = col
                    ms = 4
                    style = '-'
                    if val in ['true', 'Excluded from training']:
                        ms = 12
                        style = '*'

                    _data = data[col][st:en].values
                    # process_axis(ax, _data, style='*', ms=ms, label=val, verbose=True)
                    process_axis(ax, _data, style=style, ms=ms, leg_ms=1, label=val, leg_fs=14, leg_pos='upper left',
                                 verbose=True)
            else:
                val = cols[idx]
                _data = data[cols[idx]][st:en].values
                process_axis(ax, _data, style='*', ms=10, c=colors[val], log=obs_logy, verbose=True)
                process_axis(ax, _data, style='-', ms=9,  label=val, leg_fs=14, verbose=True, log=obs_logy)

        elif idx == 0:  # first plot
            val = cols[idx]
            _data = data[val][st:en].values
            invert_yaxis = False
            if 'pcp' in cols[idx]:
                style = '-'
                invert_yaxis = True
            process_axis(ax, _data, style='o', ms=8, c=colors[val])
            process_axis(ax, _data, style='-', ms=4,  label=val, show_xaxis=False, bottom_spine=False, leg_fs=14,
                         verbose=True, invert_yaxis=invert_yaxis)

        idx += 1

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()


def first_nan_from_end(ar):
    """ 
    This function finds index for first nan from the group which is present at the end of array.
    [np.nan, np.nan, 0,2,3,0,3, np.nan, np.nan, np.nan, np.nan] >> 7
    [np.nan, np.nan, 1,2,3,0, np.nan, np.nan, np.nan] >> 6
    [0,2,3,0,3] >> 5
    [np.nan, np.nan, 0,2,3,0,3] >> 7    
    """
    last_non_zero = 0
    
    for idx, val in enumerate(ar[::-1]):
        if ~np.isnan(val):  # val >= 0:
            last_non_zero = idx
            break
    return ar.shape[0] - last_non_zero


class BatchGenerator(object):
    """
    :param data: `ndarray`, input data.
    :param args: a dictionary containing values of parameters depending upon method used.
    :param method: str, default is 'many_to_one', if many_to_one, then following keys are expected in 
                   dictionary args.
            :lookback: `int`, sequence length, number of values LSTM will see at time `t` to make prediction at `t+1`.
            :in_features: `int`, number of columns in `data` starting from 0 to be considered as input
            :out_features: `int`, number of columns in `data` started from last to be considred as output/prediction.
            :trim_last_batch: bool, if True, last batch will be ignored if that contains samples less than `batch_size`.
            :norm: a dictionary which contains scaler object with which to normalize x and y data. We use separate
            scalers for x and y data. Keys must be `x_scaler` and `y_scaler`.
            :batch_size:
            :step: step size in input data
            :min_ind: starting point from `data`
            :max_ind: end point from `data`
            :future_y_val: number of values to predict
    """
    
    def __init__(self, data, batch_size, args, method='many_to_one', verbose=2):
        
        self.data = data
        self.batch_size = batch_size
        self.args = args
        self.method = method
        self.verbose = verbose
        self.ignoriert_am_anfang = None
        self.ignoriert_am_ende = None
        self.no_of_batches = None
    
    def __len__(self):
        return self.args['min_ind'] - self.args['max_ind']
    
    def many_to_one(self, predef_interval=None):

        many_to_one_args = {'lookback': 'required',
                            'in_features': 'required',
                            'out_features': 'required',
                            'min_ind': 0,
                            'max_ind': self.data.shape[0],
                            'future_y_val': 'required',
                            'step': 1,
                            'norm': None,
                            'trim_last_batch': True}

        for k, v in many_to_one_args.items():
            if v == 'required':
                if k not in self.args:
                    raise ValueError('for {} method, value of {} is required'.format(self.method, k))
                else:
                    many_to_one_args[k] = self.args[k]
            else:
                if k in self.args:
                    many_to_one_args[k] = self.args[k]

        lookback = many_to_one_args['lookback']
        in_features = many_to_one_args['in_features']
        out_features = many_to_one_args['out_features']
        self.min_ind = many_to_one_args['min_ind']
        self.max_ind = many_to_one_args['max_ind']
        future_y_val = many_to_one_args['future_y_val']
        step = many_to_one_args['step']
        norm = many_to_one_args['norm']
        trim_last_batch = many_to_one_args['trim_last_batch']

        # selecting the data of interest for x and y    
        x_data = self.data[self.min_ind:self.max_ind, 0:in_features]
        y_data = self.data[self.min_ind:self.max_ind, -out_features:].reshape(-1, out_features)

        if norm is not None:
            x_scaler = norm['x_scaler']
            y_scaler = norm['y_scaler']
            x_data = x_scaler.fit_transform(x_data)
            y_data = y_scaler.fit_transform(y_data)

        # container for keeping x and y windows. A `windows` is here defined as one complete set of data at
        # one timestep.
        x_wins = np.full((x_data.shape[0], lookback, in_features), np.nan, dtype=np.float32)
        y_wins = np.full((y_data.shape[0], out_features), np.nan)

        # creating windows from X data
        st = lookback*step - step  # starting point of sampling from data
        for j in range(st, x_data.shape[0]-lookback):
            en = j - lookback*step
            indices = np.arange(j, en, -step)
            ind = np.flip(indices)
            x_wins[j, :, :] = x_data[ind, :]

        # creating windows from Y data
        for i in range(0, y_data.shape[0]-lookback):
            y_wins[i, :] = y_data[i+lookback, :]

        """removing trailing nans"""
        first_nan_at_end = first_nan_from_end(y_wins[:, 0])  # first nan in last part of data, start skipping from here
        y_wins = y_wins[0:first_nan_at_end, :]
        x_wins = x_wins[0:first_nan_at_end, :]
        if self.verbose > 1:
            print('first nan from end is at: {}, x_wins shape is {}, y_wins shape is {}'
                  .format(first_nan_at_end, x_wins.shape, y_wins.shape))

        """removing nans from start"""
        y_val = st-lookback + future_y_val
        if st > 0:
            x_wins = x_wins[st:, :]
            y_wins = y_wins[y_val:, :]

        if self.verbose > 1:
            print("""shape of x data: {} \nshape of y data: {}""".format(x_wins.shape, y_wins.shape))

            print("""{} values are skipped from start and {} values are skipped from end in output array"""
                  .format(st, x_data.shape[0]-first_nan_at_end))
        self.ignoriert_am_anfang = st
        self.ignoriert_am_ende = x_data.shape[0]-first_nan_at_end

        pot_samples = x_wins.shape[0]

        if self.verbose > 1:
            print('potential samples are {}'.format(pot_samples))

        residue = pot_samples % self.batch_size
        if self.verbose > 1:
            print('residue is {} '.format(residue))
        self.residue = residue

        samples = pot_samples - residue
        if self.verbose > 1:
            print('Actual samples are {}'.format(samples))
        self.samples = samples

        if predef_interval is None:
            interval = np.arange(0, samples + self.batch_size, self.batch_size)
            if self.verbose > 1:
                print('Potential intervals: {}'.format(interval))
            interval = np.append(interval, pot_samples)
            interval = check_interval_validity(interval, x_wins.shape[0])

        else:
            interval = predef_interval
            # x_wins may have shape smaller than than data, and interval was based on data, so we need to make sure
            # that values in interval do not exceed length of x_wins
            interval = check_interval_validity(list(interval), x_wins.shape[0])
            if trim_last_batch:
                inf_bat_sz = np.unique(np.diff(np.array(predef_interval)))  # inferred batch size
            else:
                # last batch will be of different size
                inf_bat_sz = np.unique(np.diff(np.array(predef_interval[0:-1])))
                self.last_bat_sz = predef_interval[-1] - predef_interval[-2]

            if len(inf_bat_sz) > 1:
                raise ValueError("predefined array must have constant steps")
            if inf_bat_sz != self.batch_size:
                raise ValueError("Inferred batch size from predefined array is not equal to batch size defined")

        # nterval = np.unique(interval)
        if self.verbose > 1:
            print('Actual interval: {} '.format(interval))

        if trim_last_batch:
            no_of_batches = len(interval)-2
        else:
            no_of_batches = len(interval)-1 

        if no_of_batches == 0:
            no_of_batches = 1

        if self.verbose > 0:
            print('Number of batches are {} '.format(no_of_batches))
        self.no_of_batches = no_of_batches

        # code for generator
        gen_i = 1
        while 1:

            for b in range(no_of_batches):
                st = interval[b]
                en = interval[b + 1]
                x_batch = x_wins[st:en, :, :]
                y_batch = y_wins[st:en]

                gen_i += 1

                yield x_batch, y_batch


def check_and_initiate_batch(generator_object, _batch_generator, verbose=1, raise_error=True,
                             skip_batch_with_no_labels=False):
    """

    :param generator_object:
    :param _batch_generator:
    :param verbose:
    :param raise_error:
    :param skip_batch_with_no_labels: if True, then those batches which have no labels will be ignored altogether. In
      such a case, argument `raise_error` will be rendered useless. This should be used only to generate training data,
      if we want to optimize batch size, because this option will allow us to have flexible batch size.
    :return:
    """
    x_batch, mask_y_batch = next(_batch_generator)
    y_of_interest = mask_y_batch[np.where(mask_y_batch > 0.0)]
    if verbose > 0:
        print('x_batch shape: ', x_batch.shape,
              'y_batch_shape: ', mask_y_batch.shape,
              'shape of y of 1st interest:', y_of_interest.shape)

    no_of_batches = generator_object.no_of_batches
    no_of_batches_recalc = no_of_batches
    batch_size = x_batch.shape[0]
    lookback = x_batch.shape[1]
    in_features = x_batch.shape[2]
    out_features = mask_y_batch.shape[1]

    if hasattr(generator_object, 'last_bat_sz') or skip_batch_with_no_labels:
        # batch size is variable so one array of all batches can not be constructed
        x_batches = []     # not predefining length of this list because it can vary if a batch contains no labels
        y_batches = []
    else:
        x_batches = np.full((no_of_batches, batch_size, lookback, in_features), np.nan)
        y_batches = np.full((no_of_batches, batch_size, out_features), np.nan)

    # this for loop is so that next time first batch in for loop is really the first batch
    for i in range(no_of_batches - 1):
        _, _ = next(_batch_generator)

    total_bact_samples = {}
    for i in range(out_features):
        total_bact_samples[i] = 0

    if verbose > 0:
        print('\n*********************************')
        print('batch ', 'Non zeros')

    skip_this_batch = False
    for i in range(no_of_batches):

        mask_x_batch, mask_y_batch = next(_batch_generator)

        if verbose > 0:
            print(i, end='      ')

        for out_feat in range(out_features):

            a, = np.where(mask_y_batch[:, out_feat] > 0.0)
            non_zeros = a.shape[0]
            total_bact_samples[out_feat] += non_zeros

            if verbose > 1:
                print(non_zeros, mask_y_batch[a].reshape(-1, ), end=' ')
            elif verbose > 0:
                print(non_zeros, end=' ')
            if non_zeros < 1:
                if skip_batch_with_no_labels:
                    skip_this_batch = True        # we want to skip this batch and
                    no_of_batches_recalc -= 1     # total number of batches will be reduced by 1.
                else:
                    if raise_error:
                        raise ValueError('At minibatch {} exists where all labels are missing'.format(i))

        if skip_this_batch:
            if verbose > 0:
                print('skipping batch no {}'.format(i))
        else:
            if hasattr(generator_object, 'last_bat_sz') or skip_batch_with_no_labels:
                x_batches.append(mask_x_batch)
                y_batches.append(mask_y_batch)
            else:
                x_batches[i, :] = mask_x_batch
                y_batches[i, :] = mask_y_batch

        # next batch should not be skipped by default
        skip_this_batch = False

        if verbose > 0:
            print('')
    if verbose > 0:
        print('total observations: ', total_bact_samples)
        print('*********************************\n')

    return x_batches, y_batches, no_of_batches_recalc


def check_interval_validity(interval, check_against):

    interval2 = interval.copy()
    for val in interval:
        if val > check_against:
            interval2.remove(val)
    return interval2


def generate_event_based_batches(data, batch_size, args, predef_intervals, verbosity=1,
                                 raise_error=True,
                                 skip_batch_with_no_labels=False):
    args = deepcopy(args)
    events = len(predef_intervals)

    x_batches = []
    y_batches = []
    no_of_batches = 0

    for event_intvl in predef_intervals:
        # no value in interval should be greater than length of data
        event_intvl = check_interval_validity(event_intvl, data.shape[0])

        event_intvl = np.array(event_intvl)  # convert to numpy array if not already

        if not isinstance(event_intvl, np.ndarray):
            raise ValueError("Predefined arrays for each event must be numpy array")

        if len(np.unique(np.diff(event_intvl))) > 1:
            args['trim_last_batch'] = False

        event_generator = BatchGenerator(data, batch_size, args, verbose=verbosity)
        _gen = event_generator.many_to_one(predef_interval=event_intvl)

        event_x_batches,\
            event_y_batches,\
            _no_of_batches = check_and_initiate_batch(event_generator,
                                                      _gen, verbosity,
                                                      raise_error=raise_error,
                                                      skip_batch_with_no_labels=skip_batch_with_no_labels)

        no_of_batches += _no_of_batches

        for x_batch, y_batch in zip(event_x_batches, event_y_batches):
            x_batches.append(x_batch)
            y_batches.append(y_batch)

    if verbosity > 0:
        for x_batch, y_batch in zip(x_batches, y_batches):
            print(x_batch.shape, y_batch.shape)

    return x_batches, y_batches, no_of_batches


def nan_to_num(array, outs, replace_with=0.0):
    array = array.copy()
    y = np.array(array[:, -outs:], dtype=np.float32)
    y = np.nan_to_num(y, nan=replace_with)
    array[:, -outs:] = y
    return array


def maybe_create_path(prefix=None, path=None):
    if path is None:
        save_dir = dateandtime_now()
        model_dir = os.path.join(os.getcwd(), "models")

        if prefix:
            model_dir = os.path.join(model_dir, prefix)

        save_dir = os.path.join(model_dir, save_dir)
    else:
        save_dir = path

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def check_min_loss(epoch_loss_array, batch_loss_array, _epoch, func1, func, _ps, _save_fg, to_save=None):
    current_epoch_loss = np.mean(batch_loss_array)

    if len(epoch_loss_array) > 1:
        min_loss = func1(epoch_loss_array)
    else:
        min_loss = current_epoch_loss

    min_loss_epoch = None

    if func(current_epoch_loss, min_loss):
        min_loss_epoch = _epoch
        _ps = _ps + "    {:10.5f} ".format(current_epoch_loss)

        if to_save is not None:
            _save_fg = True
            # saver.save(session, save_path = save_path,  global_step=epoch)
    else:
        _ps = _ps + "              "

    epoch_loss_array.append(current_epoch_loss)

    return _ps, min_loss_epoch, _save_fg


def normalize_data(dataset):
    # normalizing data and making batches
    # container for normalized input data
    dataset_n = np.full(dataset.shape, np.nan)
    all_scalers = OrderedDict()

    for dat in range(dataset.shape[1]):
        value = dataset[:, dat]
        val_scaler = MinMaxScaler(feature_range=(0, 1))
        val_norm = val_scaler.fit_transform(value.reshape(-1, 1))
        dataset_n[:, dat] = val_norm.reshape(-1, )
        all_scalers[str(dat) + '_scaler'] = val_scaler

    return dataset_n, all_scalers


def plot_loss(train_loss_array, test_loss_array, _name, _path):
    _fig, (ax1) = plt.subplots(1, sharex='all')
    _fig.set_figheight(6)

    process_axis(ax1, train_loss_array, '-', label='Training ' + _name, leg_pos="upper left", leg_fs=12)

    process_axis(ax1, test_loss_array, '-', label='Validation ' + _name, leg_pos="upper left", leg_fs=12, y_label=_name,
                 x_label="Epochs", xtp_c='no', ytp_c='no')
    plt.subplots_adjust(wspace=0.05, hspace=0.01)
    plt.savefig(_path + "/loss_" + _name, dpi=300, bbox_inches='tight')
    plt.close(_fig)


def copy_check_points(_saved_epochs, _path):
    ch_points_to_copy = np.unique(list(_saved_epochs.values()))
    cp_to_copy = ch_points_to_copy[ch_points_to_copy != 0]  # removing zeros
    cp_to_copy = cp_to_copy[cp_to_copy != 1]
    cp_copied = []
    for chpt in cp_to_copy:
        data_file = os.path.join(os.getcwd(), "check_points-" + str(chpt) + ".data-00000-of-00001")
        if os.path.exists(data_file):
            cp_copied.append(chpt)
            copyfile(data_file, os.path.join(_path, "check_points-" + str(chpt) + ".data-00000-of-00001"))
            idx_file = os.path.join(os.getcwd(), "check_points-" + str(chpt) + ".index")
            copyfile(idx_file, os.path.join(_path, "check_points-" + str(chpt) + ".index"))
            meta_file = os.path.join(os.getcwd(), "check_points-" + str(chpt) + ".meta")
            copyfile(meta_file, os.path.join(_path, "check_points-" + str(chpt) + ".meta"))

    return [int(cp) for cp in cp_copied]


def get_errors(true_data, predicted_data, monitor):
    errors = {}
    _er = FindErrors(true_data.reshape(-1, ), predicted_data.reshape(-1, ))
    all_errors = _er.all_methods
    for err in all_errors:
        errors[err] = float(getattr(_er, err)())

    for er_name in monitor:
        print(er_name, errors[er_name])
    return errors


def dateandtime_now():
    jetzt = datetime.datetime.now()
    jahre = str(jetzt.year)
    month = str(jetzt.month)
    if len(month) < 2:
        month = '0' + month
    tag = str(jetzt.day)
    if len(tag) < 2:
        tag = '0' + tag
    date = jahre + month + tag

    stunde = str(jetzt.hour)
    if len(stunde) < 2:
        stunde = '0' + stunde
    minute = str(jetzt.minute)
    if len(minute) < 2:
        minute = '0' + minute

    save_dir = date + '_' + stunde + str(minute)
    return save_dir


def save_config_file(config, _path, from_config=False):

    if from_config:
        suffix = dateandtime_now()
        config_file = _path + "/config" + suffix + ".json"
    else:
        config_file = _path + "/config.json"

    with open(config_file, 'w') as fp:
        json.dump(config, fp, sort_keys=True, indent=4)


def plot_scatter(true, pred, _name, searborn=True):
    if searborn:
        regplot_using_searborn(true, pred, _name)
    else:
        fig, ax = plt.subplots(1)
        set_fig_dim(fig, 8, 6)

        ax.scatter(true, pred)
        ax.set_ylabel('Predicted ', fontsize=14, color='k')
        ax.set_xlabel('Observed ', fontsize=14, color='k')
        ax.tick_params(axis="x", which='major', labelsize=12, colors='k')
        ax.tick_params(axis="y", which='major', labelsize=12, colors='k')
        plt.savefig(_name, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close(fig)


def regplot_using_searborn(true, pred, _name):
    # https://seaborn.pydata.org/generated/seaborn.regplot.html
    sns.regplot(x=true, y=pred, color="g")
    plt.savefig(_name, dpi=300, bbox_inches='tight')
    plt.close('all')


def plot_bact_points(true, pred, _name):
    fig, ax = plt.subplots(1)
    set_fig_dim(fig, 14, 6)
    ax.set_title("Model Performance", fontsize=18)

    process_axis(ax, true, style='b.', c='b', ms=5)
    process_axis(ax, true, style='b-', c='b', ms=2, label="True", leg_fs=12, leg_ms=4)
    process_axis(ax, pred, style='r*', c='r', ms=6, y_label="ARGs(copies/mL)", yl_fs=14)
    process_axis(ax, pred, style='r-', c='r', ms=2, label='Predicted', leg_fs=12, leg_ms=4, y_label="ARGs(copies/mL)", yl_fs=14,
                 x_label="No. of Observations", xl_fs=14)
    plt.savefig(_name, dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_pred_where_obs_available(_true, _pred):
    _true_idx, = np.where(_true > 0.0)  # because np.where will sprout a tuple

    y_pred_avail_only = _pred[_true_idx]
    y_true_avail_only = _true[_true_idx]

    return y_true_avail_only, y_pred_avail_only


def validate_dictionary(dictionary, keys, name):
    for k in keys:
        if k not in dictionary.keys():
            raise KeyError('dictionary {} does not have key {}'.format(name, k))


def plot_single_output(df, _name):
    fig, (ax1, ax2) = plt.subplots(2, sharex='all')
    set_fig_dim(fig, 14, 6)
    ax1.set_title("Model Performance", fontsize=18)

    process_axis(ax1, df['pcp_mm'], style='b-', c='g', ms=5, invert_yaxis=True, bottom_spine=False, show_xaxis=False)

    process_axis(ax2, df['true'], style='b.', c='b', ms=5)
    # process_axis(ax, df['Used for training'], style='b-', c='b', ms=2, label="Used for training", leg_fs=12, leg_ms=4)
    # process_axis(ax, df['Excluded from training'], style='b*', c='b', ms=5)
    # process_axis(ax, df['Excluded from training'], style='b-', c='b', ms=2, label='Used for testing', leg_fs=12,
    #              leg_ms=4)
    # process_axis(ax, df['Prediction'], style='r.', c='r', ms=6, y_label="MPN", yl_fs=14)
    process_axis(ax2, df['Prediction'], style='r-', c='r', ms=2, label='Predicted', leg_fs=12, leg_ms=4, y_label="MPN",
                 yl_fs=14, top_spine=False,
                 x_label="No. of Observations", xl_fs=14)

    plt.savefig(_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
