import cv2
import flask
import numpy as np
from io import BytesIO
from requests_toolbelt import MultipartEncoder

from pose import client
from dab.model import Model, DABNET_MODEL_PATH
from data.dataset import Dataset
from data.preprocessing import extract_pose_features

app = flask.Flask(__name__)
# Load model
model = Model.load(DABNET_MODEL_PATH)
dataset = Dataset(csv_only=True)

@app.route('/predict', methods=['POST'])
def predict():
    # Receive and validate input image
    img_file = flask.request.files.get('image')
    if img_file is None:
        return ('No image sent', 400)
    img_buffer = BytesIO(img_file.read())
    img_array = np.asarray(img_buffer.getbuffer(), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Send to Posenet
    features = extract_pose_features([image])
    annotated_image = client.request_annotations(image)

    # preform prediction with model
    prediction = model.predict(features)[0]
    prediction = dataset.lookup_label(prediction)

    # encode annotated image
    is_success, annotated_img_buffer = cv2.imencode(".jpg", annotated_image)
    annotated_img_buffer = annotated_img_buffer.tostring()
    assert is_success
    print(prediction)

    response_data = {
        'annotated_image': ('annotated_image', annotated_img_buffer, 'image/jpeg'),
        'result': prediction,
    }

    multipart_data = MultipartEncoder(fields=response_data)
    return flask.Response(
        multipart_data.to_string(),
        mimetype=multipart_data.content_type
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
