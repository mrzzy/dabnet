#
# dataset.py
# dabnet
# dab dataset 
#

import os
import numpy as np
import pandas as pd
import cv2
from pose.client import request_pose
from data.preprocessing import encode_one_hot
from multiprocessing import Pool, cpu_count

# Dataset path constants
PATH = "data"
IMG_PATH = os.path.join(PATH, "images")
META_PATH = os.path.join(PATH, "meta.csv")


class Dataset:
    def __init__(self):
        images, labels, self.label_index = self.load()
    

    # Loads the dataset from disk
    # Return images, labels, label_index
    def load(self):
        # read dataset metadata
        df = pd.read_csv(META_PATH, 
                         dtype={"img_path": str, "label": "category"})
        labels =  df.loc[:, "label"].cat.codes.values
        index = df.loc[:, "label"].cat.categories
        label_index = dict([ (label_idx, label_str) for label_idx, label_str in \
                            enumerate(index) ])

        # read dataset images
        img_paths =  df.loc[:, "img_path"].values
        images = [ cv2.imread(p) for p in img_paths ]
        
        return images, labels, label_index
    
    # Lookup human readable version of the given integer label
    # returns the human readable label
    def lookup_label(self, label_idx):
        return self.label_index[label_idx]
        
        

if __name__ == "__main__":
    dataset =  Dataset()
    print(dataset.lookup_label(0))
