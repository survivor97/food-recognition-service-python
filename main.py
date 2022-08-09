from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import cv2
from skimage import io

app = Flask(__name__)
api = Api(app)

def bad_request(message):
    response = jsonify({'message': message})
    response.status_code = 404
    return response

class FoodRecognitionClass(Resource):
    def post(self):

        try:
            imgPath = request.json['path']
        except:
            print('Posted json object does not match!')
            return bad_request('Posted json object does not match!')

        try:
            m = hub.KerasLayer('model')

            cake_url = imgPath
            labelmap_url = "resources/aiy_food_V1_labelmap.csv"
            input_shape = (224, 224)

            image = np.asarray(io.imread(cake_url), dtype="float")
            image = cv2.resize(image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)

            # Scale values to [0, 1].
            image = image / image.max()

            # The model expects an input of (?, 224, 224, 3).
            images = np.expand_dims(image, 0)

            output = m(images)
            predicted_index = output.numpy().argmax()
            classes = list(pd.read_csv(labelmap_url)["name"])

            return {
                "img-path": imgPath,
                "prediction": classes[predicted_index]
            }
        except:
            print('Error in the processing of the requested image link!')
            return bad_request('Error in the processing of the requested image link!')

api.add_resource(FoodRecognitionClass, "/recognize-food")

if __name__ == "__main__":
    app.run(debug=True)