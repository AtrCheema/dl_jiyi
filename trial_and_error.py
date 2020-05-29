
from optimization import objective_func
import numpy as np
from collections import OrderedDict

# BatchSize = 24
# intervals = {
#         0: [i for i in range(0, 156, BatchSize)],
#         1: [i for i in range(136, 407, BatchSize)],
#         2: [i for i in range(412, 546, BatchSize)],
#         3: [i for i in range(509, 580, BatchSize)],
#         4: [i for i in range(533, 660, BatchSize)],
#         5: [i for i in range(633, 730, BatchSize)],
#         6: [i for i in range(730, 831, BatchSize)],
#         7: [i for i in range(821, 971, BatchSize)],
#         8: [i for i in range(941, 1071, BatchSize)],
#         9: [i for i in range(1125, 1200, BatchSize)],
#         10: [i for i in range(1172, 1210, BatchSize)],
#         11: [i for i in range(1196, 1240, BatchSize)],
#         12: [i for i in range(1220, 1317, BatchSize)],
#         13: [i for i in range(1292, 1335, BatchSize)],
#         14: [i for i in range(1316, 1447, BatchSize)]
# }
# all_train_intervals = {}
# for i in range(25):
#      all_train_intervals[i] = np.sort(random.sample(intervals.keys(), k=11))

all_train_intervals = {
          0: [0,  1,  2,  3,  4,  5,  8, 11, 12, 13, 14],
          # 1: [0,  1,  2,  3,  4,  7,  9, 10, 11, 13, 14],
          # 2: [0,  2,  3,  4,  6,  7,  9, 11, 12, 13, 14],
          # 3: [0,  2,  3,  4,  5,  7,  8,  9, 12, 13, 14],
          # 4: [0,  1,  2,  3,  4,  5,  6,  7,  8, 11, 13],
          # 5: [0,  1,  2,  3,  4,  7,  8, 10, 11, 12, 13],
          # 6: [0,  1,  3,  4,  5,  7,  9, 10, 12, 13, 14],
          # 7: [0,  3,  4,  6,  7,  8,  9, 10, 12, 13, 14],
          # 8: [1,  2,  3,  4,  5,  6,  7, 10, 11, 13, 14],
          # 9: [1,  2,  3,  4,  5,  7,  9, 10, 11, 13, 14],
          # 10: [0,  3,  4,  5,  6,  8,  9, 11, 12, 13, 14],
          # 11: [0,  2,  3,  5,  7,  8,  9, 10, 11, 12, 14],
          # 12: [0,  1,  2,  3,  5,  6,  7,  8,  9, 12, 14],
          # 13: [0,  1,  2,  3,  8,  9, 10, 11, 12, 13, 14],
          # 14: [0,  1,  2,  4,  6,  7,  8, 10, 12, 13, 14],
          # 15: [0,  3,  4,  5,  6,  7,  8, 10, 12, 13, 14],
          # 16: [0,  1,  2,  3,  5,  6,  7,  9, 10, 11, 14],
          # 17: [0,  1,  2,  3,  4,  6,  7,  9, 11, 12, 13],
          # 18: [0,  1,  2,  3,  4,  6,  7,  8, 11, 13, 14],
          # 19: [0,  1,  2,  5,  6,  9, 10, 11, 12, 13, 14],
          # 20: [2,  4,  6,  7,  8,  9, 10, 11, 12, 13, 14],
          # 21: [1,  2,  3,  5,  6,  7,  8,  9, 10, 12, 14],
          # 22: [1,  4,  5,  6,  7,  8,  9, 10, 11, 12, 14],
          # 23: [0,  1,  3,  4,  7,  8,  9, 11, 12, 13, 14],
          # 24: [0,  1,  4,  5,  6,  7, 10, 11, 12, 13, 14]
}


items = tuple(all_train_intervals.values())
train_intervals = OrderedDict((tuple(x), x) for x in items).values()


for val in train_intervals:

    er1, er2 = objective_func(pot_tr_intervals=val)
