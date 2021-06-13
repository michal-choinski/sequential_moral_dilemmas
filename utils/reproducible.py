import os
import random as rn
import numpy as np
from tensorflow import set_random_seed

def set_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    rn.seed(seed)