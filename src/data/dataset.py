#
# dataset.py
# dabnet
# dab dataset 
#

import os
import cv2
import numpy as np
import pandas as pd
import hashlib
from shutil import rmtree
from data.preprocessing import encode_one_hot, extract_pose_features

# Dataset path constants
PATH = "data"
IMG_PATH = os.path.join(PATH, "images")
CACHE_PATH = os.path.join(PATH, "cache")
CACHE_DATA_PATH = os.path.join(CACHE_PATH, "data.npz")
CACHE_SIZE_PATH = os.path.join(CACHE_PATH, "size.txt")
CACHE_CHECKSUM_PATH = os.path.join(CACHE_PATH, "checksum.sha1")
META_PATH = os.path.join(PATH, "meta.csv")

class Dataset:
    # n_limit - limits dataset size
    # use_cache - load dataset from cache if possible
    # meta_only - only load metadata not entire dataset
    # NOTE: use_cache and n_limit are mutually exclusive
    def __init__(self, n_limit=None, use_cache=True, csv_only=False):
        self.load_meta(n_limit)
        if not csv_only: self.load(n_limit, use_cache=True)

    # Load metadata from doisk
    def load_meta(self, n_limit):
        # read dataset metadata
        df = pd.read_csv(META_PATH,
                         dtype={"img_path": str, "label": "category"})
        if n_limit: df = df.sample(n_limit)
        # extract dataset laabels
        self.labels =  df.loc[:, "label"].cat.codes.values
        index = df.loc[:, "label"].cat.categories
        self.label_index = dict([ (label_idx, label_str) for label_idx, label_str in \
                            enumerate(index) ])

        # read dataset images pahts
        self.img_paths =  df.loc[:, "img_path"].values

    # Loads the dataset from disk
    # limits dataset to  n_limit entries
    # uses dataset cache if use_cache is true
    # NOTE: use_cache and n_limit are mutually exclusive
    def load(self, n_limit, use_cache=True):
        assert not (n_limit and use_cache == True)

        # read dataset images
        self.images = [ cv2.imread(p) for p in self.img_paths ]

        # Prepare the dataset images and labels for machine learning
        self.label_vectors = encode_one_hot(self.labels)

        # load features from cache if possible
        if use_cache and self.check_cache():
            self.load_cache()
        else:
            self.features = extract_pose_features(self.images)
            self.save_cache() # save features to cache if possible
        

    # Check if the dataset can be loaded from cache
    # Returns true if can load from cache
    def check_cache(self):
        if not os.path.exists(CACHE_PATH): return False
        elif not os.path.exists(CACHE_CHECKSUM_PATH): return False
        else:
            # read and verify checksum to determine cache validity
            with open(META_PATH, "rb") as meta_file:
                # compute checksum for meta file
                meta_checksum =  hashlib.sha1(meta_file.read()).digest()

            # read cache checksum
            with open(CACHE_CHECKSUM_PATH, "rb") as f:
                cache_checksum = f.read()
            if not meta_checksum == cache_checksum: return False
        return True

    # Cache the dataset to disk to speed up loading
    # overwrites existing cache if present
    def save_cache(self):
        # setup cache directory
        rmtree(CACHE_PATH, ignore_errors=True)
        os.makedirs(CACHE_PATH)

        # write cache data to disk
        with open(CACHE_DATA_PATH, "wb") as f:
            np.savez(f, features=self.features)

        # write cache checksum
        with open(META_PATH, "rb") as meta_file:
            cache_checksum =  hashlib.sha1(meta_file.read()).digest()
        with open(CACHE_CHECKSUM_PATH, "wb") as f:
            f.write(cache_checksum)

    # Load the dataset in the cach
    def load_cache(self):
        assert self.check_cache()
        # read cache data from disk
        with open(CACHE_DATA_PATH, "rb") as f:
            self.features = np.load(f)["features"]

    # Lookup human readable version of the given label
    # returns the human readable label
    def lookup_label(self, label_vec):
        print(label_vec.shape)
        label_idx = np.argmax(label_vec)
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
    output = dataset.outputs[-1]
