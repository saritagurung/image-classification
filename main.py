import numpy as np
#import tensorflow as tf
from flask import Flask, render_template
from flask import request, jsonify
from PIL import Image

img_height = 180
img_width =180
class_names = ["daisy", "dandelion", "roses","sunflowers","tulips"]

new_model = tf.keras.models.load_model('saved_model/initial_model')

app = Flask(__name__)
@app.route("/")
def home_page():
     return render_template('index.html')

@app.route("/image", methods = ['GET', 'POST'])
def predict_image():
     data = request.files
     print(data)
     img = Image.open(data["file"])
     img = np.array(img)
     img = tf.image.resize(img, (img_height,img_width))
     img_array = tf.expand_dims(img,0) #create a batch file

     predictions = new_model.predict(img_array)
     score = tf.nn.softmax(predictions[0])

     return jsonify(
          {
               "res": "This image most likely belongs to {} with a {:.2f} percent confidence."\
        .format(class_names[np.argmax(score)], 100 * np.max(score))
          }
     )