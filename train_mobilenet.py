#load tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
#load other packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Path to all images
path_to_images = os.path.join('C:\\Users\\cordovam\\Desktop\\project 2')

#Path to training and validation image sets
train_dir = os.path.join(path_to_images, 'project_train')
val_dir = os.path.join(path_to_images, 'project_val')

#Raths to individual class folders
train_0_dir = os.path.join(train_dir, '0')
train_1_dir = os.path.join(train_dir, '1')
val_0_dir = os.path.join(val_dir, '0')
val_1_dir = os.path.join(val_dir, '1')

#how many total images are in each set. These counts are used to calculate steps per epoch when we fit the model
total_train = len(os.listdir(train_0_dir))+ len(os.listdir(train_1_dir))
total_val = len(os.listdir(val_0_dir))+ len(os.listdir(val_1_dir))

#Parameters
batch_size = 256
epochs = 50
#Image size
IMG_HEIGHT = 96
IMG_WIDTH = 96


##############PICK TRAINING AUGMENTATION
####no augmentation
# train_image_generator = ImageDataGenerator(rescale=1./255) 

#### Generator for our training data with data augmentation
train_image_generator = ImageDataGenerator(rescale=1./255, 
                                            horizontal_flip = True,
                                            vertical_flip = True)
                                            #rotation_range=15, zoom_range=0.1) 
###################


# Generator for our validation data
validation_image_generator = ImageDataGenerator(rescale=1./255) 

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode="binary")

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=val_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode="binary")

sample_training_images, labs = next(train_data_gen)
#############PICK A NETWORK
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                               include_top=False,
                                               weights='imagenet')

#########################

global_average_layer = tf.keras.layers.GlobalAveragePooling2D(
    data_format='channels_last')

prediction_layer = tf.keras.layers.Dense(1)


model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch= total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps= total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

#performance data to save as csv
filename = 'mobilenet.csv'
pd.DataFrame(data = np.column_stack((acc, val_acc, loss, val_loss)), 
             columns = ['acc', 'val_acc', 'loss', 'val_loss'], 
             index = list(epochs_range)).to_csv(os.path.join(path_to_images,filename))

#plots that we want to look at and save?, but we'll make better ones in R
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