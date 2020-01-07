#
# preprocessing.py
# dabnet
# data preprocessing
#

import numpy as np
import pandas as pd
from multiprocessing import pool, cpu_count

from pose.client import request_pose

# One encode the given inputs
def encode_one_hot(inputs):
    n_values = max(inputs) - min(inputs) + 1
    encoding = np.zeros((len(inputs), n_values))
    encoding[np.arange(len(inputs)), inputs] = 1
    
    return encoding

# Flatten the np arrays in the given nested list of pose features
def flatten_features(features):
    # Flatten the numpy array feature
    def flatten_inner(features):
        return [ np.ravel(feat) for feat in features ]
    features = [ flatten_inner(feat_set) for feat_set in features ]

    # stack each featre in feature set together
    features = [ np.concatenate(feat_set) for feat_set in features ]

    return features

# Extract flattened features for the given images
# Returns extracted flattened fewatures
def extract_pose_features(images):
    process_pool = pool.Pool(cpu_count())
    #pose_feats = [ request_pose(img) for img in images ]
    pose_feats = process_pool.map(request_pose, images, chunksize=2)
    flat_pose_feats = flatten_features(pose_feats)

    return flat_pose_feats
