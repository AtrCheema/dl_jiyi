
from utils import maybe_create_path, plot_loss, copy_check_points
from utils import get_pred_where_obs_available, get_errors
from utils import plot_bact_points, plot_scatter
from utils import save_config_file, do_plot

import numpy as np
from collections import OrderedDict
import pandas as pd


def post_process(data_conf,  x_batches, y_batches, test_dataset,
                 model, saved_epochs, _path, scalers,  full_args,
                 losses, verbose=1):

    handle_losses(losses, _path)

    saved_unique_cp = copy_check_points(saved_epochs, _path)
    # print(type(saved_unique_cp), saved_unique_cp.shape, saved_unique_cp)

    all_errors = {}
    neg_predictions = {}
    for ep in saved_unique_cp:
        sub_path = _path + '/' + str(ep)
        maybe_create_path(path=sub_path)

        check_point = "check_points-" + str(ep)

        x_data, test_y_pred, test_y_true = model.run_check_point(check_point=check_point,
                                                                 x_batches=x_batches,
                                                                 y_batches=y_batches,
                                                                 data_set=test_dataset,
                                                                 scalers=scalers)

        # create a separate folder for each target and save its relevent data in that folder
        for idx, out in enumerate(data_conf['out_features']):
            out_path = sub_path + '/' + out
            maybe_create_path(path=out_path)

            _test_y_pred = test_y_pred[:, idx]
            _test_y_true = test_y_true[:, idx]

            negative_predictions = np.sum(np.array(_test_y_pred) < 0, axis=0)
            if negative_predictions > 0:
                print("Warning, {} Negative bacteria predictions found".format(negative_predictions))
            neg_predictions[str(ep)+'_'+out] = int(negative_predictions)

            if negative_predictions > 0:
                _test_y_true = _test_y_true.copy()
            else:
                _test_y_true = np.where(_test_y_true > 0.0, _test_y_true, np.nan)

            test_y_true_avail, test_y_pred_avail = get_pred_where_obs_available(_test_y_true, _test_y_pred)

            test_errors = get_errors(test_y_true_avail, test_y_pred_avail, data_conf['monitor'])

            all_errors[str(ep)+'_'+out] = test_errors

            print(_test_y_pred.shape, _test_y_true.shape)

            if verbose > 1:
                for i, j in zip(_test_y_pred, _test_y_true):
                    print(i, j)

            plot_bact_points(test_y_true_avail, test_y_pred_avail, out_path + "/bact_points")

            plot_scatter(test_y_true_avail, test_y_pred_avail, out_path + "/scatter")

            ndf = pd.DataFrame()
            ndf['true'] = _test_y_true
            ndf[out] = _test_y_pred
            # ndf['true_avail'] = test_y_true_avail
            # ndf['pred_avail'] = test_y_pred_avail

            do_plot(ndf, ndf.columns, save_name=out_path + '/' + str(out), obs_logy=True, single_ax_plots=['true', out])

        config = OrderedDict()
        config['comment'] = 'use point source pollutant data along with best model from grid search'
        config['nn_config'] = model.nn_conf
        config['data_config'] = data_conf
        # config['train_errors'] = train_errors
        config['test_errors'] = all_errors
        config['test_sample_idx'] = 'test_idx'
        config['start_time'] = model.config['start_time'] if 'start_time' in model.config else " "
        config['end_time'] = model.config['end_time'] if 'end_time' in model.config else " "
        config["saved_epochs"] = saved_epochs
        config['train_time'] = ""
        config['final_comment'] = """ """
        config['negative_predictions'] = neg_predictions

        save_config_file(config, _path)

        return all_errors


def handle_losses(losses, _path):

    if losses is not None:
        train_losses = losses[0]
        val_losses = losses[1]
        pd.DataFrame.from_dict(train_losses).to_csv(_path + '/train_losses.txt')
        pd.DataFrame.from_dict(val_losses).to_csv(_path + '/val_losses.txt')

        # plot losses
        for er in train_losses.keys():
            plot_loss(train_losses[er], val_losses[er], er, _path)
