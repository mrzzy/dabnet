#
# collector.py
# dabnet
# data collector
#

import cv2
import os
import pandas as pd
import numpy as np
from pose.client import request_annotations
from data import dataset
from PIL import Image

# record the given frame with the given label in the dataset
# df is dataframe that records the metadata in dataset 
def record_frame(df, frame, label):
    if not os.path.exists(dataset.IMG_PATH): os.makedirs(dataset.IMG_PATH)

    # commit frame to disk
    img_idx = len(df)
    img_name = "{}.jpg".format(img_idx)
    img_path = os.path.join(dataset.IMG_PATH, img_name)
    cv2.imwrite(img_path, frame)
    
    # record metadata into dataframe
    df.loc[img_idx] = { "img_path": img_path, "label": label }
    dataset.PATH
    print("{} written, marked as {}".format(img_name, label))

# Annotate the given frame with human pose annotates
def annotate_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
        
    annotated_image = request_annotations(image)
    annotated_img_rgb = np.asarray(annotated_image)
    annotated_frame = cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR)
    
    return annotated_frame
    
# setup dataframe
columns = [ "img_path", "label" ]
df_path = dataset.META_PATH
if os.path.exists(df_path):
    df = pd.read_csv(df_path)  
else:
    df = pd.DataFrame(columns=columns)

# Read and record images and label to build dataset
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    if not ret: break
    annotated_frame = annotate_frame(frame)
    cv2.imshow("Controls: q - quit, d - dab, n - not dab", annotated_frame)

    key = cv2.waitKey(1)
    if key%256 == ord('q'):
        # q pressed
        break # stop processing
    elif key%256 == ord('d'): # record a dab
        record_frame(df, frame, "dab")
    elif key%256 == ord('n'): # record not a dab
        record_frame(df, frame, "notdab")
    else:
        continue
camera.release()
cv2.destroyAllWindows()
    
# Commit metadata dataframe
if not os.path.exists(dataset.PATH): os.makedirs(dataset.PATH)
df.to_csv(df_path, index=False)
