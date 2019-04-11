#
# dataset.py
# dabnet
# dab dataset 
#

import os
import numpy as np
import pandas as pd
import cv2
from data.preprocessing import encode_one_hot, extract_pose_features

# Dataset path constants
PATH = "data"
IMG_PATH = os.path.join(PATH, "images")
CACHE_PATH = os.path.join(PATH, "cache")
META_PATH = os.path.join(PATH, "meta.csv")

class Dataset:
    # n_limit - limits dataset size
    def __init__(self, n_limit=None):
        self.images, self.labels, self.label_index = self.load(n_limit)
        self.prepare()

        
    # Loads the dataset from disk
    # limits dataset to  n_limit entries
    # Return images, labels, label_index
    def load(self, n_limit):
        # read dataset metadata
        df = pd.read_csv(META_PATH, 
                         dtype={"img_path": str, "label": "category"})
        if n_limit: df = df.sample(n_limit)
        # extract dataset laabels
        labels =  df.loc[:, "label"].cat.codes.values
        index = df.loc[:, "label"].cat.categories
        label_index = dict([ (label_idx, label_str) for label_idx, label_str in \
                            enumerate(index) ])

        # read dataset images
        img_paths =  df.loc[:, "img_path"].values
        images = [ cv2.imread(p) for p in img_paths ]
        
        return images, labels, label_index

    # Prepare the dataset images and labels for machine learning
    def prepare(self):
        self.features = extract_pose_features(self.images)
        self.label_vectors = encode_one_hot(self.labels)
        
    # Lookup human readable version of the given integer label
    # returns the human readable label
    def lookup_label(self, label_idx):
        return self.label_index[label_idx]
    
    # ML inputs generated from dataset
    @property
    def inputs(self):
        return self.features

    # ML outputs generated from dataset
    @property
    def outputs(self):
        return self.label_vectors

if __name__ == "__main__":
    dataset =  Dataset()
