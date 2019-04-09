#
# src/pose/server.py
# dabnet
# posenet server
#

import os
import h5py
import hashlib
from uuid import uuid4
from pose import api
from pose.extraction import extract_pose, annotate_pose
from flask import Flask, request, redirect, url_for, send_file

app = Flask("pose.server")

# Constants
# temporary storage path for uploaded files
UPLOAD_PATH = "/tmp/dabnet/posenet/uploaded"
# storage path for computed pose features
FEATURES_PATH = "/tmp/dabnet/posenet/features"
# storage path for annotaed images
ANNOTATED_IMG_PATH = "/tmp/dabnet/posenet/annotated"

## Utilities
# Returns a hash of the contents of the file speicified by file path
def hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()

# save the uploaded file specificed by key in files
# returns the path of the saved file
def save_uploaded_file(key, files):
    file = request.files[key]
    path = "{}/{}.jpg".format(UPLOAD_PATH, str(uuid4()))
    file.save(path)
    
    return path

# package the given features dictionary dict as a h5py file 
# writes the h5py file in the path specified by path
def package_features_h5(path, features_dict):
    with h5py.File(path, "w") as f:
        for key, feature in features_dict.items():
            f.create_dataset(key, data=feature)

## Server routes
# Display server status: used to check if the server is running properly 
@app.route("/")
def route_status():
    return "Posenet server is alive and running!", 200

# API POST: submit multipart form data consisting of an image
# Performs posenet human pose feature extraction on the image
# Returns an hdf5 file containing the extracted features
@app.route(api.FEATURES_ROUTE, methods=["POST"])
def route_features():
    prefix = "[API]:[{}]: ".format(api.FEATURES_ROUTE)

    # Save the uploaded image file
    img_path = save_uploaded_file("target_image", request.files)
    
    # Hash the image to determine if features hash already been computed
    job_id = hash(img_path)
    print(prefix, "recieved feature extraction request for image with hash: ",
          job_id)
    
    # only compute features if not already existing
    feats_path = "{}/{}.hdf5".format(FEATURES_PATH, job_id)
    if not os.path.exists(feats_path):
        # Perform feature extraction
        print(prefix,  "performing pose extraction...", end="")
        pose_scores, keypoint_scores, keypoint_points = extract_pose(img_path)
        print("Done")
        
        # Package features in  hdf5  file
        feats_path = "{}/{}.hdf5".format(FEATURES_PATH, job_id)
        package_features_h5(feats_path, {
            api.POSE_SCORE_FEATURE: pose_scores,
            api.KEYPOINT_SCORE_FEATURE: keypoint_scores,
            api.KEYPOINT_POINTS_FEATURE: keypoint_points
        })

    return send_file(feats_path)


# API POST: submit multipart form data consisting of an image
# Performs posenet human pose detection
# annotates detected keypoints onto the image
@app.route(api.ANNOTATION_ROUTE, methods=["POST"])
def route_annotation():
    prefix = "[API]:[{}]: ".format(api.FEATURES_ROUTE)

    # Save the uploaded image file
    img_path = save_uploaded_file("target_image", request.files)
    
    # Hash the image to determine if features hash already been computed
    job_id = hash(img_path)
    print(prefix, "recieved annotation  for image with hash: ",
          job_id)

    # only compute features if not already existing
    annotated_img_path = "{}/{}.jpg".format(ANNOTATED_IMG_PATH, job_id)
    if not os.path.exists(annotated_img_path):
        # Perform annotation of image
        print(prefix,  "performing annotation ...", end="")
        annotate_pose(img_path, out_img_path=annotated_img_path)
        print("Done")

    return send_file(annotated_img_path)

        
if __name__ == "__main__":
    # setup directories
    if not os.path.exists(UPLOAD_PATH): os.makedirs(UPLOAD_PATH)
    if not os.path.exists(FEATURES_PATH): os.makedirs(FEATURES_PATH)
    if not os.path.exists(ANNOTATED_IMG_PATH): os.makedirs(ANNOTATED_IMG_PATH)
        
    app.run(host="0.0.0.0", port=api.SERVER_PORT)
