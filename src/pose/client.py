#
# src/pose/client.py
# dabnet
# posenet client
#

import h5py
import os
import requests
import numpy as np
from pose import api
from PIL import Image
from io import BytesIO
from tempfile import NamedTemporaryFile

# Constants
SERVER_URL = f"http://localhost:{api.SERVER_PORT}"

# Request pose features for given pillow image
# NOTE: only accepts jpg images
# Returns features: pose_scores, keypoint_scores, keypoint_points
def request_pose(image):
    # write image to buffer for upload
    img_buffer = BytesIO()
    image.save(img_buffer, format="jpeg")

    # request server for pose feature for image
    files = {"target_image": img_buffer.getvalue()}
    response = requests.post(f"{SERVER_URL}{api.POSE_ROUTE}", 
                             files=files)
    
    # extract features from server response
    feats_buffer = BytesIO(response.content)
    with h5py.File(feats_buffer, "r") as f:
        print("h5 keys:", f.keys())
        pose_scores = np.asarray(f[api.POSE_SCORE_FEATURE])
        keypoint_scores = np.asarray(f[api.KEYPOINT_SCORE_FEATURE])
        keypoint_points = np.asarray(f[api.KEYPOINT_POINTS_FEATURE])
    
    return pose_scores, keypoint_scores, keypoint_points 
    
# Request pose annotations for the image at the given path
# NOTE: only accepts jpg images
# Returns image with annotations
def request_annotations(img_path):
    # request server for pose annotations  for image
    files = {"target_image": open(img_path, "rb")}
    response = requests.post(f"{SERVER_URL}{api.POSE_ROUTE}", 
                             files=files)

if __name__ == "__main__":
    print("Requesting  server for human pose features for camp david statue...")
    features = request_pose(Image.open("david.jpg"))
    print(features)
