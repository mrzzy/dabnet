#
# collector.py
# dabnet
# data collector
#

import cv2
import os
import pandas as pd
from pose.client import request_annotations

DATASET_PATH = "data"
DATASET_IMG_PATH = os.path.join(DATASET_PATH, "images")

# record the given frame with the given label in the dataset
# df is dataframe that records the metadata in dataset 
def record_frame(df, frame, label):
    if not os.path.exists(DATASET_IMG_PATH): os.makedirs(DATASET_IMG_PATH) 

    # commit frame to disk
    img_idx = len(df)
    img_name = "{}.jpg".format(img_idx)
    img_path = os.path.join("images", img_name)
    cv2.imwrite(img_path, frame)
    
    # record metadata into dataframe
    df.loc[img_idx] = { "img_path": img_path, "label": label }

    print("{} written, marked as {}".format(img_name, label))

# setup dataframe
columns = [ "img_path", "label" ]
df_path = os.path.join(DATASET_PATH, "meta.csv")
if os.path.exists(df_path):
    df = pd.read_csv(df_path)  
else:
    df = pd.DataFrame(columns=columns)

# Read and record images and label to build dataset
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()

    cv2.imshow("Controls: q - quit, d - dab, n - not dab", frame)
    if not ret: break

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
if not os.path.exists(DATASET_PATH): os.makedirs(DATASET_PATH)
df.to_csv(df_path, index=False)
