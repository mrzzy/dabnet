import flask
from requests_toolbelt import MultipartEncoder

from pose import client
from dab import Model, DABNET_MODEL_PATH

app = flask.Flask(__name__)
# Load model
model = Model.load(DABNET_MODEL_PATH)
dataset = Dataset()

@app.route('/predict', methods=['POST'])
def predict():
    # Receive and validate input image
    image = flask.request.files.get('image')
    if image is None:
        return ('No image sent', 400)

    # Send to Posenet
    features = [client.request_pose(image)]
    annotated_image = client.request_annotations(image)

    # preform prediction with model
    prediction = model.predict(features)[0]
    prediction = dataset.lookup_label(prediction)

    response_data = {
        'annotated_image': ('annotated_image', annotated_image, 'image/jpeg'),
        'result': prediction,
    }

    multipart_data = MultipartEncoder(fields=response_data)
    return flask.Response(
        multipart_data.to_string(),
        mimetype=multipart_data.content_type
    )


if __name__ == '__main__':
    app.run(debug=True)
