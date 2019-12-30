#
# model.py
# dabnet
# model.py
#

import os
import math
import pickle
import numpy as np
from data.dataset import Dataset
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Path constants
MODELS_PATH = "models"
DABNET_MODEL_PATH = os.path.join(MODELS_PATH, "model.pickle")

# Dabnet model
class Model:
    def __init__(self, scaler=None, model=None):
        self.scaler = scaler if scaler else StandardScaler()
    
        # Create model if does not already present 
        if not model: 
            M = 40
            self.backend_model = RandomForestClassifier(n_estimators=M,
                                                     max_features=math.floor(M ** 0.5),
                                                     n_jobs=-1)
            self.has_fit = False
        else:
            self.backend_model = model
            self.has_fit = True
    
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
    
    # Save the model to disk at the given path
    # Overwrite existing model if already exists
    def save(self, path):
        # setup models directory
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname): os.makedirs(dirname)
        
        # write model to disk
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.backend_model,
                "scaler": self.scaler
            }, f)
    
    @classmethod
    # Loads and returns the model at the given path
    def load(cls, path):
        with open(path, "rb") as f:
            contents = pickle.load(f)

        model = contents["model"]
        scaler = contents["scaler"]

        return cls(scaler=scaler, model=model)

if __name__ == "__main__":
    dataset = Dataset()
    model = Model()
    model.fit(dataset.inputs, dataset.outputs)
    predictions = model.predict(dataset.inputs)

    model.save(DABNET_MODEL_PATH)
    model = None
    model = Model.load(DABNET_MODEL_PATH)
    predictions = model.predict(dataset.inputs)
