#
# dataset.py
# dabnet
# dab dataset 
#

import os
import numpy as np
import pandas as pd
import cv2



# Dataset path constants
PATH = "data"
IMG_PATH = os.path.join(PATH, "images")
META_PATH = os.path.join(PATH, "meta.csv")

# Loads the dataset from disk
# Returns features and labels np arrays
def load():
    # read dataset metadata
    df = pd.read_csv(META_PATH)
    labels =  df.loc[:, "label"].values
    img_paths =  df.loc[:, "img_path"].values
    
    # read dataset images
    images = [ cv2.imread(p) for p in img_paths ]

if __name__ == "__main__":
    load()
