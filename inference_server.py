import cv2
# keras setup
from keras.models import load_model
from keras.preprocessing.image import img_to_array

from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def preprocess(image):
    """
    function that preprocesses and makes sure image is correct size and is gray scale
    :param image: raw input image
    :return: grayscale image of size (28,28,1) for inference
    """
    image = image.convert('L')
    image = np.asarray(image)
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    """
    inference function
    """
    # initialize the data dictionary that will be returned from the
    # view
    data = {"Success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            # image = np.asarray(image)

            # preprocess the image and prepare it for classification
            image = preprocess(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            prediction = model.predict(image)
            label = int(np.argmax(prediction))
            prob = np.max(prediction)
            data["Predictions"] = []

            r = {"Label": label, "Softmax probability": float(prob)}
            data["Predictions"].append(r)

            # indicate that the request was a success
            data["Success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    model = load_model('model.h5')
    app.run()

