#
# src/pose/client.py
# dabnet
# posenet client
#

import h5py
import os
import requests
import numpy as np
from tempfile import NamedTemporaryFile
from pose import api

# Constants
SERVER_URL = f"http://localhost:{api.SERVER_PORT}"

# Request pose features for image at the given path
# NOTE: only accepts jpg images
# Returns features: pose_scores, keypoint_scores, keypoint_points
def request_pose(img_path):
    # request server for pose feature for image
    files = {"target_image": open(img_path, "rb")}
    response = requests.post(f"{SERVER_URL}{api.POSE_ROUTE}", 
                             files=files)
    
    # save server response
    with NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(response.content)
        feats_path = f.name

    # extract features from server response
    with h5py.File(feats_path, "r") as f:
        print("h5 keys:", f.keys())
        pose_scores = np.asarray(f[api.POSE_SCORE_FEATURE])
        keypoint_scores = np.asarray(f[api.KEYPOINT_SCORE_FEATURE])
        keypoint_points = np.asarray(f[api.KEYPOINT_POINTS_FEATURE])
    
    return pose_scores, keypoint_scores, keypoint_points 
    

if __name__ == "__main__":
    print("Requesting  server for human pose features for camp david statue...")
    features = request_pose("david.jpg")
    print(features)
