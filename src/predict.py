import numpy as np
import tensorflow as tf

img_height = 180
img_width =180
class_names = ["daisy", "dandelion", "roses","sunflowers","tulips"]

new_model = tf.keras.models.load_model('saved_model/initial_model')
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = new_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(score)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."\
        .format(class_names[np.argmax(score)], 100 * np.max(score)))





