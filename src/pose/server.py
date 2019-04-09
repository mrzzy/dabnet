#
# src/pose/server.py
# dabnet
# posenet server
#

import os
import h5py
from uuid import uuid4
from pose import api
from pose.extraction import extract_pose
from flask import Flask, request, redirect, url_for, send_file

app = Flask("pose.server")

# Constants
# temporary storage path for uploaded files
UPLOAD_PATH = "/tmp/dabnet/posenet/uploaded"
# temporary storage path for computed pose features
FEATURES_PATH = "/tmp/dabnet/posenet/features"

## Server routes
# Display server status: used to check if the server is running properly 
@app.route("/")
def route_status():
    return "Posenet server is alive and running!", 200

# POST: submit multipart form data consisting of an image
# Performs posenet human pose feature extraction on the image
# Returns an hdf5 file containing the extracted features
@app.route(api.POSE_ROUTE, methods=["POST"])
def route_pose():
    prefix = "[API]:[{}]: ".format(api.POSE_ROUTE)
    job_id = str(uuid4())
    print(prefix,  "received request: assigned job id: ", job_id)

    # Save the uploaded image file
    img_file = request.files["target_image"]
    img_path = "{}/{}.jpg".format(UPLOAD_PATH, job_id)
    img_file.save(img_path)
    
    # Perform feature extraction
    print(prefix,  "performing pose extraction...", end="")
    pose_scores, keypoint_scores, keypoint_points = extract_pose(img_path)
    print("OK")
    
    # Package features in  hdf5  file
    feats_path = "{}/{}.hdf5".format(FEATURES_PATH, job_id)
    with h5py.File(feats_path, "w") as f:
        f.create_dataset(api.POSE_SCORE_FEATURE, data=pose_scores)
        f.create_dataset(api.KEYPOINT_SCORE_FEATURE, data=keypoint_scores)
        f.create_dataset(api.KEYPOINT_POINTS_FEATURE, data=keypoint_points)

    return send_file(feats_path)
    
        
if __name__ == "__main__":
    # setup directories
    if not os.path.exists(UPLOAD_PATH): os.makedirs(UPLOAD_PATH)
    if not os.path.exists(FEATURES_PATH): os.makedirs(FEATURES_PATH)
        
    app.run(host="0.0.0.0", port=api.SERVER_PORT)
