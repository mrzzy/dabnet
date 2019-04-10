import cv2

import requests
import json

#######################
with open('server_config.json') as file:
    config = json.load(file)

SERVER_URL = f"http://{config['IP']}:{config['Port']}/"

########################
cam = cv2.VideoCapture(0)
cv2.namedWindow('Capture')
cv2.namedWindow('Feature Extraction')

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    if not ret: break
    # Display frame
    cv2.imshow("Capture", frame)
    cv2.waitKey(1)

    # Send to Server
    # r = requests.post(SERVER_URL, data={'Action': 'predict'},files={'images': [frame]})
    # data = r.json()
    # annotated_images = data['annotated_images']
    # result = data['result']
    # cv2.imshow('Feature Extraction', annotated_images[0])
    # print(result[0])

cam.release()
cv2.destroyAllWindows()
