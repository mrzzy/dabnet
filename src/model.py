import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class Model:
    def __init__(self, size=(510, 510)):
        self.SIZE = size
        self.INPUT_SIZE = size[0] * size[1]

        self.model = Sequential()
        self.model.add(Dense(100, input_dim=self.INPUT_SIZE))
        self.model.add(Dense(1))
        self.model.compile(
                optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy']
        )

    def preprocess(self, features):
        data = []
        for feature in features:
            _, _, keypoint_points = feature
            coords = keypoint_points[0]
            ret = np.zeros(self.SIZE)
            for x,y in coords:
                ret[int(x),int(y)] = 1
            ret = ret.ravel()
            data.append(ret)
        return np.array(data)
    
    def train(self, features, labels, **kwargs):
        data = self.preprocess(features)
        self.model.fit(data, labels, **kwargs)
        
    def evaluate(self, features_test, labels_test, **kwargs):
        data_test = self.preprocess(features_test)
        score = self.model.evaluate(data_test, labels_test, **kwargs)
        return score

    def predict(self, features, **kwargs):
        data = self.preprocess(features)
        prediction = self.model.predict(data, **kwargs)
        return prediction

# import pickle
# with open('data', 'rb') as file:
#     data = pickle.load(file)
    
# model = Model()
# model.train([data], [1])
