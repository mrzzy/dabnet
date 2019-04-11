#
# model.py
# dabnet
# model.py
#

from data.dataset import Dataset
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
    
# Dabnet model
class Model:
    # Fit the model to the given data given as inputs, expected outputs
    def fit(inputs, outputs):
        pass
        
    # Predict the outputs for the given
    def predict(inputs): 
        pass

if __name__ == "__main__":
    dataset = Dataset(n_limit=30)
    model = Model()
    model.fit(dataset.inputs, dataset.outputs)
