
from utils import maybe_create_path
from utils import get_pred_where_obs_available, get_errors
from utils import plot_bact_points, plot_scatter
from utils import do_plot
from utils import plot_single_output

import numpy as np
import pandas as pd
import os


def make_predictions(x_batches,
                     y_batches,
                     model,
                     epochs_to_evaluate,
                     # scalers,
                     runtype,
                     save_results=False):

    all_errors = {}
    neg_predictions = {}
    for ep in epochs_to_evaluate:
        sub_path = model.path + '/' + str(ep)
        maybe_create_path(path=sub_path)

        check_point = "checkpoints-" + str(ep)

        x_data, _y_pred, _y_true = model.run_check_point(check_point=check_point,
                                                                 x_batches=x_batches,
                                                                 y_batches=y_batches,
                                                                 scalers=model.scalers[runtype])

        # create a separate folder for each target and save its relevent data in that folder
        for idx, out in enumerate(model.data_config['out_features']):
            out_path = sub_path + '/' + out + '_' + runtype
            maybe_create_path(path=out_path)

            y_pred = _y_pred[:, idx]
            y_true = _y_true[:, idx]

            negative_predictions = np.sum(np.array(y_pred) < 0, axis=0)
            if negative_predictions > 0:
                print("Warning, {} Negative bacteria predictions found".format(negative_predictions))
            neg_predictions[str(ep)+'_'+out] = int(negative_predictions)

            if negative_predictions > 0:
                y_true = y_true.copy()
            else:
                y_true = np.where(y_true > 0.0, y_true, np.nan)

            y_true_avail, y_pred_avail = get_pred_where_obs_available(y_true, y_pred)

            errors = get_errors(y_true_avail, y_pred_avail, model.data_config['monitor'])

            all_errors[str(ep)+'_'+out] = errors

            print('shapes of predicted arrays: ', y_pred.shape, y_true.shape, x_data.shape)

            if model.verbosity > 2:
                for i, j in zip(y_pred, y_true):
                    print(i, j)

            plot_scatter(y_true_avail, y_pred_avail, out_path + "/scatter")

            ndf = pd.DataFrame()

            # fill ndf with input data
            for i, inp in enumerate(model.data_config['in_features']):
                ndf[inp] = x_data[:, i]

            ndf['true'] = y_true
            ndf[out] = y_pred
            # ndf['true_avail'] = test_y_true_avail
            # ndf['pred_avail'] = test_y_pred_avail

            ndf.index = get_index(model.batches[runtype + '_index'])

            # removing duplicated values
            # TODO why duplicated values exist
            ndf = ndf[~ndf.index.duplicated(keep='first')]

            plots_on_last_axis = ['true', out]
            if runtype == 'all':

                if model.data_config['batch_making_mode'] == 'event_based':
                    train_idx = get_index(model.batches['train' + '_index'])
                    train_idx = train_idx[~train_idx.duplicated()]
                else:
                    train_tk = model.batches['train_tk_index']
                    train_tk_nz = train_tk[np.where(train_tk > 0.0)]
                    train_idx = get_index(train_tk_nz)

                # test_idx = get_index(model.batches['test' + '_index'])
                out_df = ndf[out]
                out_df = out_df[~out_df.index.duplicated()]
                ndf['train'] = ndf[out][train_idx]  # out_df[train_idx]
                # ndf['test'] = ndf[out][test_idx]
                plots_on_last_axis.append('train')

            do_plot(ndf, list(ndf.columns), save_name=out_path + '/' + str(out), obs_logy=True,
                    single_ax_plots=plots_on_last_axis)

            ndf['Prediction'] = ndf[out]
            plot_single_output(ndf, out_path + '/' + str(out) + '_single', runtype)

            plot_bact_points(ndf, out_path + "/bact_points", runtype)

            if save_results:
                fpath = os.path.join(out_path + '_' + runtype + '_results.xlsx')
                ndf.to_excel(fpath)

    return all_errors, neg_predictions


def get_index(idx_array, fmt='%Y%m%d%H%M'):
    """ converts a numpy 1d array into pandas DatetimeIndex type."""

    if not isinstance(idx_array, np.ndarray):
        raise TypeError

    return pd.to_datetime(idx_array.astype(str), format=fmt)
