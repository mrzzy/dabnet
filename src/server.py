import flask
from requests_toolbelt import MultipartEncoder

from pose import client
from model import Model


app = flask.Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Receive and validate input image
    image = flask.request.files.get('image')
    if image is None:
        return ('No image sent', 400)

    # Load model
    model = Model.load()

    # Feature Extraction
    # TODO: Feature extraction

    # Send to Posenet
    features = [client.request_pose(image)]
    annotated_image = client.request_annotations(image)

    # More Feature Extraction
    # TODO: More feature extraction

    result = model.predict(features)

    response_data = {
        'annotated_image': ('annotated_image', annotated_image, 'image/jpeg'),
        'result': result,
    }

    multipart_data = MultipartEncoder(fields=response_data)
    return flask.Response(
        multipart_data.to_string(),
        mimetype=multipart_data.content_type
    )


if __name__ == '__main__':
    app.run(debug=True)
