import numpy as cp
threshold = 2

exc_act = [[1, 3, 4],
           [2, 2.5, 6],
           [1, 1.3, 4]]

exc_act = cp.array(exc_act)

#exc_act = cp.maximum(exc_act - threshold, 0) - cp.maximum(-exc_act - threshold, 0)

#print(exc_act)


