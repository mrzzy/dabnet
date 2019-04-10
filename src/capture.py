import cv2
import requests

from . import settings


class VideoCapture:
    '''Context manager for OpenCV Video Capture object

    This context manager calls `.release()` on the video capture resource on
    exit.
    '''
    def __init__(self, *args):
        self.video_capture_args = args

    def __enter__(self):
        self.resource = cv2.VideoCapture(*self.video_capture_args)
        return self.resource

    def __exit__(self, type, value, traceback):
        self.resource.release()


class CVWindow:
    '''Context manager for OpenCV windows

    This context manager destroys the window that it creates upon exit. It
    returns the window name in `__enter__`. This means that it can be used as
    so:
    ```
    >>> with CVWindow('my window name') as window_name:
    >>>     print(window_name)
    my window name
    ```
    This makes it useful for displaying frames on the window.
    '''
    def __init__(self, window_name):
        self.name = window_name

    def __enter__(self):
        cv2.namedWindow(self.name)

        return self.name

    def __exit__(self, type, value, traceback):
        cv2.destroyWindow(self.name)


def send_frame(url, frame):
    file_payload = {
        'frame': frame
    }

    response = requests.post(url, files=file_payload)

    # Ensure that the status code is 200
    if response.status_code != 200:
        raise ValueError(f'Response status code was {response.status_code},'
                         ' not 200.')

    return response.json()


with VideoCapture(0) as cam, \
        CVWindow('Capture') as capture_window, \
        CVWindow('Feature Extraction') as feature_window:
    while True:
        ret, frame = cam.read()

        if not ret:
            break

        cv2.imshow(capture_window, frame)
        response_data = send_frame(settings.SERVER_URL, frame)

        # Show the annotated image
        annotated_image = response_data['annotated_image']
        cv2.imshow(feature_window, annotated_image)

        result = response_data['result']
        print(f'Is a dab? {result}')

        # Quit if 'q' is pressed
        c = cv2.waitKey(1)
        if 'q' == chr(c & 255):
            break
