
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from stolera_scheduler  import Stolera
from dilera_scheduler import Dilera
from tensorflow.keras.optimizers.schedules import *

print(tf.__version__)

dataset = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])


#
# STOLERA 
################
learning_rate = 1
sigma = 5
seed = 2.0
lr_schedule = Dilera(learning_rate, sigma, seed)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)


# Grab an image from the test dataset.
img = test_images[1]

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))


predictions_single = probability_model.predict(img)

print(predictions_single)


np.argmax(predictions_single[0])