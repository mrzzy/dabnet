import cv2
from PIL import Image

import requests
import json

#######################
with open('server_config.json') as file:
    config = json.load(file)

SERVER_URL = f"http://{config['IP']}:{config['Port']}/"

#######################
def cvt_cv2_to_pil_image(cv2_image):
    img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    return img_pil

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

    # Convert to PIL Image
    img = cvt_cv2_to_pil_image(frame)

    # Send to Server
    r = requests.post(SERVER_URL, files={'image': img})
    

cam.release()
cv2.destroyAllWindows()
