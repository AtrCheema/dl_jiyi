

import numpy as np
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize, BayesSearchCV

from optimization import objective_func


with open('opt_results.txt', 'w') as f:
    f.write('Hyper OPT Results')


dim_batch_size = Categorical(categories=[24, 32, 40], name='batch_size')
dim_lookback = Integer(low=6, high=16, prior='uniform', name='lookback')
dim_learning_rate = Real(low=1e-7, high=1e-6, prior='uniform', name='lr')
dim_lstm_units = Categorical(categories=[100, 120, 140, 160, 180, 200], name='lstm_units')
dim_act_f = Categorical(categories=['relu', 'tanh'], name='act_f')

dim_rel_hum = Categorical(categories=[True, False], name='rel_hum')
dim_mslp_hpa = Categorical(categories=[True, False], name='mslp_hpa')
dim_air_p_hpa = Categorical(categories=[True, False], name='air_p_hpa')
dim_wind_speed_mps = Categorical(categories=[True, False], name='wind_speed_mps')
dim_wind_dir_deg = Categorical(categories=[True, False], name='wind_dir_deg')
dim_air_temp_c = Categorical(categories=[True, False], name='air_temp_c')
dim_sal_psu = Categorical(categories=[True, False], name='sal_psu')
dim_wat_temp_c = Categorical(categories=[True, False], name='wat_temp_c')


default_values = [32, 8, 1.e-6, 100, 'relu',
                  True, True, True, True,
                  True, True, True, True]

dimensions = [dim_batch_size, dim_lookback, dim_learning_rate, dim_lstm_units, dim_act_f,
              dim_rel_hum, dim_mslp_hpa, dim_air_p_hpa, dim_wind_speed_mps,
              dim_wind_dir_deg, dim_air_temp_c, dim_sal_psu, dim_wat_temp_c]


@use_named_args(dimensions=dimensions)
def fitness(batch_size, lookback, lr, lstm_units, act_f,

            rel_hum, mslp_hpa, air_p_hpa, wind_speed_mps,
            wind_dir_deg, air_temp_c, sal_psu, wat_temp_c
            ):

    in_features = ['pcp_mm', 'tide_cm']

    if rel_hum:
        in_features.append('rel_hum')
    if mslp_hpa:
        in_features.append('mslp_hpa')
    if air_p_hpa:
        in_features.append('air_p_hpa')
    if wind_speed_mps:
        in_features.append('wind_speed_mps')
    if wind_dir_deg:
        in_features.append('wind_dir_deg')
    if air_temp_c:
        in_features.append('air_temp_c')
    if sal_psu:
        in_features.append('sal_psu')
    if wat_temp_c:
        in_features.append('wat_temp_c')

    msg = """\nusing in_features {}, batch_size {}, lookback {} lr {} lstm_units {} activation {}
          """.format(in_features, batch_size, lookback, lr, lstm_units, act_f)
    print(msg)

    error, pred_er = objective_func(in_features, BatchSize=batch_size,
                                    lookback=int(lookback),
                                    lr=lr,
                                    lstm_units=int(lstm_units),
                                    act_f=act_f)

    msg = """\nwith in_features {}, batch_size {}, lookback {} lr {} lstm_units {} activation {} val loss is {}
          pred_error is {}
          """.format(in_features, batch_size, lookback, lr, lstm_units, act_f, error, pred_er)
    print(msg)
    with open('opt_results.txt', 'a') as fp:
        fp.write(msg)

    return error


search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',   # Expected Improvement.
                            n_calls=20,
                            # acq_optimizer='auto',
                            x0=default_values)
