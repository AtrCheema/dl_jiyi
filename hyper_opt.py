

from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_evaluations, plot_objective, plot_convergence
import matplotlib.pyplot as plt
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

x0 = [[24, 8, 1e-06, 64, 64, 'relu', 'relu'],
      [4, 8, 8.57823924362294e-07, 256, 128, 'relu', 'tanh'],
      [4, 10, 9.81108863840166e-07, 128, 256, 'tanh', 'tanh'],
      [4, 9, 3.4551412878533e-07, 128, 64, 'relu', 'relu'],
      [4, 13, 8.44712825411554e-07, 64, 256, 'relu', 'tanh'],
      [12, 7, 2.44384438243528e-07, 128, 256, 'relu', 'relu'],
      [4, 13, 1.41993807355947e-07, 256, 128, 'tanh', 'relu'],
      [12, 15, 6.57185982042514e-07, 256, 128, 'tanh', 'tanh'],
      [12, 9, 7.07159900283542e-07, 64, 256, 'relu', 'relu'],
      [12, 8, 7.49095140845912e-07, 128, 256, 'relu', 'relu'],
      [24, 15, 7.22883503638063e-07, 128, 64, 'tanh', 'relu'],
      [12, 12, 5.31609175915916e-07, 128, 128, 'tanh', 'tanh'],
      [4, 14, 3.37099406661644e-07, 64, 128, 'tanh', 'tanh'],
      [24, 13, 2.22878767501831e-07, 128, 64, 'relu', 'relu'],
      [24, 13, 6.09557872233098e-07, 64, 64, 'relu', 'tanh'],
      [24, 9, 8.26246525639526e-07, 64, 128, 'relu', 'tanh'],
      [24, 14, 1.6411280752881e-07, 128, 256, 'relu', 'relu'],
      [12, 6, 2.48743423049466e-07, 128, 256, 'relu', 'tanh'],
      [4, 8, 7.61671356958047e-07, 64, 64, 'tanh', 'tanh'],
      [12, 13, 6.90826895333767e-07, 128, 128, 'relu', 'tanh'],
      [4, 6, 3.95552163955227e-07, 64, 128, 'relu', 'relu'],
      [12, 9, 3.86307487961487e-07, 64, 128, 'tanh', 'tanh'],
      [4, 9, 4.73339970991662e-07, 64, 128, 'tanh', 'relu'],
      [4, 16, 3.82258594944288e-07, 64, 128, 'relu', 'relu'],
      [4, 14, 2.224664090280367e-07, 256, 256, 'relu', 'tanh'],
      [24, 7, 8.779479047730352e-07, 64, 64, 'tanh', 'relu'],
      [4, 9, 8.310194792194281e-07, 64, 64, 'tanh', 'relu'],
      [24, 14, 6.218785513683412e-07, 64, 128, 'tanh', 'tanh'],
      [12, 8, 4.235069930360078e-07, 64, 256, 'tanh', 'relu'],
      [12, 13, 6.444174288648524e-07, 64, 256, 'tanh', 'relu'],
      [24, 10, 4.988194837531326e-07, 128, 128, 'relu', 'tanh'],
      [4, 8, 5.99649892900211e-07, 256, 128, 'relu', 'tanh'],
      [12, 9, 6.442503481478026e-07, 256, 128, 'tanh', 'relu'],
      [12, 7, 4.3853300384718174e-07, 256, 256, 'relu', 'tanh'],
      [24, 13, 7.430071572900693e-07, 128, 256, 'tanh', 'tanh'],
      [4, 11, 8.041070753649774e-07, 64, 256, 'relu', 'relu'],
      [12, 16, 9.037822421353196e-07, 256, 64, 'relu', 'relu'],
      [4, 16, 9.72210836228375e-07, 256, 256, 'relu', 'relu'],
      [12, 15, 1.5533513625388349e-07, 256, 256, 'relu', 'tanh'],
      [12, 16, 9.876065562693863e-07, 128, 128, 'relu', 'relu'],
      [24, 15, 9.729828806664692e-07, 256, 256, 'relu', 'relu']
      ]

y0 = [0.0120664938076383,
      0.0099889421899323,
      0.0122246645454894,
      0.0110434665964558,
      0.00910369585498869,
      0.0121004709470164,
      0.0120045482432307,
      0.0110613512377188,
      0.00904330900799553,
      0.00884245230941708,
      0.0112166185997176,
      0.0126982177421477,
      0.0113047106608894,
      0.0148594742888784,
      0.0129839958897829,
      0.011458032098498,
      0.0146922101360788,
      0.0125968475161283,
      0.0134384712413929,
      0.00982668323007441,
      0.0119480865398657,
      0.0127818186719722,
      0.012848206847,
      0.00979403273066961,
      0.00909251250848254,
      0.0142351983145787,
      0.0123455550685891,
      0.012715084,
      0.0113518334520334,
      0.0103590950836159,
      0.0105588439227688,
      0.0100212642018797,
      0.0117484616254041,
      0.009546419462197,
      0.0130167519137606,
      0.00824617592430144,
      0.008297773775904222,
      0.00820267864795182,
      0.011933804250471606,
      0.00863359152113567,
      0.008686147468450376
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
                            n_calls=15,
                            # acq_optimizer='auto',
                            n_random_starts=0,
                            x0=x0, y0=y0)


_ = plot_evaluations(search_result)
plt.savefig('evaluations', dpi=400, bbox_inches='tight')
plt.show()

_ = plot_objective(search_result)
plt.savefig('objective', dpi=400, bbox_inches='tight')
plt.show()


_ = plot_convergence(search_result)
plt.savefig('convergence', dpi=400, bbox_inches='tight')
plt.show()
