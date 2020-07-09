

from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

from optimization import objective_func


with open('opt_results.txt', 'w') as f:
    f.write('Hyper OPT Results')


dim_batch_size = Categorical(categories=[4, 12, 24], name='batch_size')
dim_lookback = Integer(low=6, high=16, prior='uniform', name='lookback')
dim_learning_rate = Real(low=1e-7, high=1e-6, prior='uniform', name='lr')
dim_lstm_units = Categorical(categories=[64, 128, 256], name='lstm_units')
dim_filters = Categorical(categories=[64, 128, 256], name='filters')
dim_lstm_act = Categorical(categories=['relu', 'tanh'], name='lstm_act')
dim_cnn_act = Categorical(categories=['relu', 'tanh'], name='cnn_act')

default_values = [24, 8, 1e-6, 64, 64, 'relu', 'relu']

x0 = [[24, 8, 0.000001, 64, 64, 'relu', 'relu'],
      [4, 8, 8.57823924362294E-07, 256, 128, 'relu', 'tanh'],
      [4, 10, 9.81108863840166E-07, 128, 256, 'tanh', 'tanh'],
      [4, 9, 3.4551412878533E-07, 128, 64, 'relu', 'relu'],
      [4, 13, 8.44712825411554E-07, 64, 256, 'relu', 'tanh'],
      [12, 7, 2.44384438243528E-07, 128, 256, 'relu', 'relu'],
      [4, 13, 1.41993807355947E-07, 256, 128, 'tanh', 'relu'],
      [12, 15, 6.57185982042514E-07, 256, 128, 'tanh', 'tanh'],
      [12, 9, 7.07159900283542E-07, 64, 256, 'relu', 'relu'],
      [12, 8, 7.49095140845912E-07, 128, 256, 'relu', 'relu'],
      [24, 15, 7.22883503638063E-07, 128, 64, 'tanh', 'relu'],
      [4, 10, 9.46446465869748E-07, 256, 64, 'relu', 'relu'],
      [12, 12, 5.31609175915916E-07, 128, 128, 'tanh', 'tanh'],
      [4, 14, 3.37099406661644E-07, 64, 128, 'tanh', 'tanh'],
      [24, 13, 2.22878767501831E-07, 128, 64, 'relu', 'relu'],
      [24, 13, 6.09557872233098E-07, 64, 64, 'relu', 'tanh'],
      [24, 9, 8.26246525639526E-07, 64, 128, 'relu', 'tanh'],
      [24, 14, 1.6411280752881E-07, 128, 256, 'relu', 'relu'],
      [12, 6, 2.48743423049466E-07, 128, 256, 'relu', 'tanh'],
      [4, 8, 7.61671356958047E-07, 64, 64, 'tanh', 'tanh'],
      [12, 13, 6.90826895333767E-07, 128, 128, 'relu', 'tanh'],
      [4, 6, 3.95552163955227E-07, 64, 128, 'relu', 'relu'],
      [12, 9, 3.86307487961487E-07, 64, 128, 'tanh', 'tanh'],
      [4, 9, 4.73339970991662E-07, 64, 128, 'tanh', 'relu'],
      [4, 16, 3.82258594944288E-07, 64, 128, 'relu', 'relu'],
      [4, 14, 2.224664090280367e-07, 256, 256, 'relu', 'tanh']
      ]

y0 =[0.00920511622617095,
     0.0099889421899323,
     0.0102887384107124,
     0.00921816164527481,
     0.00903409363141813,
     0.00918360464199022,
     0.0113024329852946,
     0.0109891876400843,
     0.00829568464459678,
     0.00884245230941708,
     0.0111169885469324,
     0.0104709727463242,
     0.0122328034303983,
     0.0110896422733222,
     0.0124894805658208,
     0.0101559529076209,
     0.0100199053370585,
     0.0117938020995654,
     0.00955357893040959,
     0.0134384712413929,
     0.00982668323007441,
     0.0105506220510067,
     0.0125536215682475,
     0.011727702676112,
     0.00826858703462918,
     0.009084224987776989
     ]

dimensions = [dim_batch_size, dim_lookback, dim_learning_rate, dim_lstm_units,
              dim_filters, dim_lstm_act, dim_cnn_act]


@use_named_args(dimensions=dimensions)
def fitness(batch_size, lookback, lr, lstm_units, lstm_act, cnn_act, filters):

    error = objective_func(batch_size, lookback, lr, lstm_units, lstm_act,
                           cnn_act,
                           filters)

    msg = """\nwith batch_size {}, lookback {} lr {} lstm_units {}
     filters {} lstm_activation {} cnn activation {} val loss is {}
          """.format(batch_size, lookback, lr, lstm_units, filters, lstm_act,
                     cnn_act, error)
    print(msg)
    with open('opt_results.txt', 'a') as fp:
        fp.write(msg)

    return error


search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',   # Expected Improvement.
                            n_calls=40,
                            # acq_optimizer='auto',
                            x0=x0, y0=y0)

from skopt.plots import plot_evaluations
from skopt.plots import plot_objective
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence

_ = plot_evaluations(search_result)
plt.savefig('evaluations', dpi=400, bbox_inches='tight')
plt.show()

_ = plot_objective(search_result)
plt.savefig('objective', dpi=400, bbox_inches='tight')
plt.show()


_ = plot_convergence(search_result)
plt.savefig('convergence', dpi=400, bbox_inches='tight')
plt.show()
