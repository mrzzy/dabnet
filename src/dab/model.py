#
# model.py
# dabnet
# model.py
#

import math
import numpy as np
from data.dataset import Dataset
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
    
# Dabnet model
class Model:
    def __init__(self, scaler=None, model=None):
        self.scaler = scaler if scaler else StandardScaler()
    
        # Create model if does not already present 
        M = 40
        if not model: model = RandomForestClassifier(n_estimators=M,
                                                     max_features=math.floor(M ** 0.5),
                                                     n_jobs=-1)
        self.backend_model = model
        self.has_fit = False
        
    
    # Preprocess the given inputs for machine learning
    def preprocess(self, inputs):
        scaled_inputs = self.scaler.transform(inputs)
        return scaled_inputs
         
    
    # Fit the model to the given data given as inputs, expected outputs
    # Returns r2 score of model
    def fit(self, inputs, outputs, split=0.3, verbose=True):
        ## Preprocess data
        # Split dataset into test and train subsets
        train_inputs, test_inputs, train_outputs, test_outputs = \
            train_test_split(inputs, outputs, test_size=0.3, shuffle=True)
        if verbose: 
            print("train on {}, test on {}".format(len(train_inputs), 
                                                   len(test_inputs)))
    
        # preprocess inputs
        self.scaler.fit(train_inputs)
        train_inputs = self.preprocess(train_inputs)
        test_inputs = self.preprocess(test_inputs)
    
        # fit model to data
        self.backend_model.fit(train_inputs, train_outputs)
        if verbose:
            print("train score:", self.backend_model.score(train_inputs, train_outputs))
            print("test score:", self.backend_model.score(test_inputs, test_outputs))
        self.has_fit = True

    # Predict the outputs for the given
    def predict(self, inputs):
        inputs = self.preprocess(inputs)
        predictions = self.backend_model.predict(inputs)
        
        return predictions
    
if __name__ == "__main__":
    dataset = Dataset()
    model = Model()
    model.fit(dataset.inputs, dataset.outputs)
    predictions = model.predict(dataset.inputs)
    predictions = [ dataset.lookup_label(p) for p in predictions ]
    print(predictions)
