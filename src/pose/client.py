#
# src/pose/client.py
# dabnet
# posenet client
#

import os
import cv2
import h5py
import requests
import numpy as np
from pose import api
from io import BytesIO
from tempfile import NamedTemporaryFile

# Constants
SERVER_URL = f"http://localhost:{api.SERVER_PORT}"

# Request pose features for given image np array
# Returns features: pose_scores, keypoint_scores, keypoint_points
def request_pose(image):
    # write image to buffer for upload
    is_success, img_buffer = cv2.imencode(".jpg", image)

    # request server for pose feature for image
    files = {"target_image": img_buffer}
    response = requests.post(f"{SERVER_URL}{api.FEATURES_ROUTE}", 
                             files=files)
    
    # extract features from server response
    feats_buffer = BytesIO(response.content)
    with h5py.File(feats_buffer, "r") as f:
        pose_scores = np.asarray(f[api.POSE_SCORE_FEATURE])
        keypoint_scores = np.asarray(f[api.KEYPOINT_SCORE_FEATURE])
        keypoint_points = np.asarray(f[api.KEYPOINT_POINTS_FEATURE])
    
    return pose_scores, keypoint_scores, keypoint_points 
    
# Request pose annotations for the image np array
# Annotates image by drawing pose features on image
# Returns image (np array) with annotations
def request_annotations(image):
    # write image to buffer for upload
    is_success, upload_buffer = cv2.imencode(".jpg", image)
    assert is_success

    # request server for pose annotations  for image
    files = {"target_image": upload_buffer}
    response = requests.post(f"{SERVER_URL}{api.ANNOTATION_ROUTE}", 
                             files=files)

    # read image from server response
    img_buffer = BytesIO(response.content)
    img_array = np.asarray(img_buffer.getbuffer(), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return image
    
if __name__ == "__main__":
    
    print("Requesting  server for human pose features for camp david statue...")
    features = request_pose(cv2.imread("david.jpg", cv2.IMREAD_COLOR))
    image = request_annotations(cv2.imread("david.jpg", cv2.IMREAD_COLOR))
    cv2.imwrite("out.jpg", image)
