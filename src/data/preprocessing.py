#
# preprocessing.py
# dabnet
# data preprocessing
#

import numpy as np
import pandas as pd
import multiprocessing


# One encode the given inputs
def encode_one_hot(inputs):
    n_values = max(inputs) - min(inputs) + 1
    encoding = np.zeros((len(inputs), n_values))
    encoding[np.arange(len(inputs)), inputs] = 1
    
    return encoding
