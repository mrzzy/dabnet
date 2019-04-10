from flask import Flask
from flask import request, jsonify

import os, json

import client
from model import Model

import numpy as np
import pickle
with open('data', 'rb') as file:
    data = pickle.load(file)

########################
with open('server_config.json') as file:
    config = json.load(file)

UPLOAD_PATH = config['Upload Path']

app = Flask(__name__)

########################
@app.route('/', methods=['POST'])
def receive_images():
    # Receive Image
    imgs = request.files.getlist('images')
    if imgs is None:
        return ('No File Sent', 404)
    
    # Get Action
    action = request.form.get('Action')
    if action is None:
        return ('No Action Indicated', 404)

    # Get Labels (if needed)
    labels = request.form.getlist('Labels')
    labels = [int(x) for x in labels]

    # Get kwargs (if needed)
    kwargs = request.form.get('kwargs')
    if not kwargs:
        kwargs = {}

    # Feature Extraction
    features = []
    annotated_imgs = []
    # for img in imgs:
    #     features.append(client.request_post(img))
    #     annotated_imgs.append(client.request_annotations(img))
    features = [data]
    print(labels)

    my_model = Model.load()

    print(action)
    if action == 'train':
        result = my_model.train(features, labels, **kwargs)
    elif action == 'evaluate':
        result = my_model.evaluate(features, labels, **kwargs)
    elif action == 'predict':
        result = my_model.predict(features, **kwargs)

    my_model.save()

    print(type(result))
    if (type(result) is np.ndarray): result = result.tolist()

    print(result)

    return jsonify({
        'annotated_images': annotated_imgs,
        'result': result
    })

if __name__ == '__main__':
    app.run(debug=True)
