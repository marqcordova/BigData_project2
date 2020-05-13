# Import TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Other useful libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__) # Print the tensorflow version

# Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# The data contains 60,000 28x28 pixel images representing one of the classes listed above.

# For compatibility with the tensorflow neural network model, the 0-255 pixel brightness values must be scaled to values on [0,1].
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the neural network model requres configuring the layers of the model, then compiling the model.

# First, we set up the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)), # Transforms images from 2-D arrays to 1-D
    keras.layers.Dense(128, activation = 'relu'), # 128 nodes in a densely-connected neural layer
    keras.layers.Dense(10) # Returns a logits array with length 10
])

# Next, we "compile" the model
model.compile(
    optimizer = 'adam', # How the model is updates based on the data it sees and its loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Measures accuracy
    metrics=['accuracy'] # Used to monitor the training and testing steps (accuracy is the fraction of images correctly classified)
)

# Finally, we train the model
# "Feed" the training data to the model
model.fit(train_images, train_labels, epochs=10)

# Now we can evaluate the accuracy of the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
