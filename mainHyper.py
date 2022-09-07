import itertools
import random
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
from tensorflow.keras.regularizers import l1,l2

def RandomSearch(ncombination=1):
  hp_opt_algorithm     =  [Adamax,Nadam,Adam]
  hp_learning_rate     =  [0.1, 1e-2, 1e-3,1e-4,1e-5]
  hp_reg_type          =  [l1, l2]
  hp_reg_values        =  [0.1, 1e-2, 1e-3, 1e-4, 1e-5]
  hyperparameters      =  [hp_opt_algorithm,hp_learning_rate,hp_reg_type,hp_reg_values]
  hp_grid              =  list(itertools.product(*hyperparameters))
  random.shuffle(hp_grid)
  for i in hp_grid:
    print(i)





# a = [["A1","A2",1],["B1","B2"],["C1","C2"],["D1","D2","D3"]]
# grid=list(itertools.product(*a))
# #random.shuffle(grid)
# print(len(grid))
#
# A=random.sample(grid,5)
#
# for i in grid:
#   print(i)