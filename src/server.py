from flask import Flask
from flask import request

import os, json

import client

########################
with open('client_server_config.json') as file:
    config = json.load(file)

UPLOAD_PATH = config['Upload Path']

app = Flask(__name__)

########################
@app.route('/', methods=['POST'])
def receive_images():
    # Receive Image
    img = request.files.get('image')
    if img is None:
        return ('No File Sent', 404)

    features = client.request_pose(img)
    annotated_img = client.request_annotations(img)

    """
    Do Feature Extraction and Stuff
    ...
    return Image, Result
    """

if __name__ == '__main__':
    app.run(debug=True)