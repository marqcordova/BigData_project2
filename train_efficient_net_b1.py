# Trains the EfficientNetB1(EFNB1) model using TensorFlow
# This script was derived from Mark's "train_mobilenet.py" script in this directory.
# added callbacks to save best model based on val_acc and RLR by 1/5 after 5 epochs
# load tensorflow

# Load dependencies, including Tensorflow and EFNB1
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import \
    EfficientNetB1  # Only available in tf-nightly as of 5.24.2020, only works with Python 3.6
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Get the current working directory and join it with the relative path to the images directory.
cwd_path = os.getcwd()  # Pretty much does the same thing as here() in R.
path_to_images = os.path.join(cwd_path + "/data")  # Appends data dir to cwd path string

# Paths to training and validation image sets
train_dir = os.path.join(path_to_images, 'project_train')
val_dir = os.path.join(path_to_images, 'project_val')

# Paths to individual class folders
train_0_dir = os.path.join(train_dir, '0')
train_1_dir = os.path.join(train_dir, '1')
val_0_dir = os.path.join(val_dir, '0')
val_1_dir = os.path.join(val_dir, '1')

# Determines the number of images in each set.
# These counts are used to calculate steps per epoch when we fit the model
total_train = len(os.listdir(train_0_dir)) + len(os.listdir(train_1_dir))
total_val = len(os.listdir(val_0_dir)) + len(os.listdir(val_1_dir))

# Parameters
batch_size = 256
epochs = 5  # Was previously 50, made this small because RIP my NVIDIA GeForce GT 750M 2 GB in this MacBook!

# Image size - These values are just the X,Y resolution of the images in the dataset
# Note - Tensorflow complained when using 96x96 and required 240x240
IMG_HEIGHT = 240
IMG_WIDTH = 240

# Prepare the training data
# Tensorflow generator that performs data augmentation. We'll see if this creates a more robust model.
train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           rotation_range=15, zoom_range=0.1)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode="binary")

sample_training_images, labs = next(train_data_gen)

# Prepare the validation data
# Generator for our validation data - Note lack of augmentation
validation_image_generator = ImageDataGenerator(rescale=1. / 255)
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=val_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode="binary")

# Choose the EFNB1 model - https://keras.io/api/applications/efficientnet/#efficientnetb1-function
base_model = tf.keras.applications.EfficientNetB1(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    pooling=None,  # Set to none because include_top is true.
    classes=1000,  # Classes = 1000 is required when using imagenet
    classifier_activation=None # Using this activation function on the top layer prevents logits from being returned.
)

# Using Global Average Pooling can improve the accuracy and efficiency of the model.
# More on this here: https://adventuresinmachinelearning.com/global-average-pooling-convolutional-neural-networks/
global_average_layer = tf.keras.layers.GlobalAveragePooling2D(
    data_format='channels_last')

# Create the prediction layer - just a regular densely-connected NN layer
prediction_layer = tf.keras.layers.Dense(1)

# Groups a linear stack of layers into a tf.keras.Model
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# Configure the model for training
# Kept this pretty much the same as Mark's code
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Add callback to save best model based on val_acc
my_callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',
                                       monitor='val_acc',
                                       verbose=1,
                                       save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=5)
]

# Fits the model based on data yielded, batch-by-batch, from the Python generator
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    callbacks=my_callbacks,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

# Save accuracy and loss for training and validation sets
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Performance data to save as csv
filename = 'efficientnetb1.csv'
pd.DataFrame(  # Create a dataframe of data with pandas
    data=np.column_stack((acc, val_acc, loss, val_loss)),
    columns=['acc', 'val_acc', 'loss', 'val_loss'],
    index=list(epochs_range)).to_csv(os.path.join(path_to_images, filename)
                                     )

# Plots that we want to look at and save?, but we'll make better ones in R
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
