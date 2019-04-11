#
# model.py
# dabnet
# model.py
#

import numpy as np
from data.dataset import Dataset
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
    
# Dabnet model
class Model:
    # Fit the model to the given data given as inputs, expected outputs
    def fit(self, inputs, outputs, split=0.3):
        # Split dataset into test and train subsets
        train_inputs, test_inputs, train_outputs, test_outputs = \
            train_test_split(inputs, outputs, test_size=0.3, shuffle=True)
        test = train_test_split(inputs, test_size=0.3)
    
        
    # Predict the outputs for the given
    def predict(self, inputs): 
        pass

if __name__ == "__main__":
    dataset = Dataset(n_limit=30)
    model = Model()
    model.fit(dataset.inputs, dataset.outputs)
