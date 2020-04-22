
from utils import maybe_create_path, plot_loss, copy_check_points
from utils import get_pred_where_obs_available, get_errors
from utils import plot_bact_points, plot_scatter
from utils import do_plot

import numpy as np
import pandas as pd


def make_predictions(data_config,  x_batches, y_batches, test_dataset,
                     model, epochs_to_evaluate, _path, scalers,  full_args,
                     verbose=1):

    all_errors = {}
    neg_predictions = {}
    for ep in epochs_to_evaluate:
        sub_path = _path + '/' + str(ep)
        maybe_create_path(path=sub_path)

        check_point = "check_points-" + str(ep)

        x_data, test_y_pred, test_y_true = model.run_check_point(check_point=check_point,
                                                                 x_batches=x_batches,
                                                                 y_batches=y_batches,
                                                                 data_set=test_dataset,
                                                                 scalers=scalers)

        # create a separate folder for each target and save its relevent data in that folder
        for idx, out in enumerate(data_config['out_features']):
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

            test_errors = get_errors(test_y_true_avail, test_y_pred_avail, data_config['monitor'])

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

        return all_errors, neg_predictions



