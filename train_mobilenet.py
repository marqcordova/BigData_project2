#trains mobilenet 
#added callbacks to save best model based on val_acc and RLR by 1/5 after 5 epochs
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

#Paths to individual class folders
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

#We're going to use flow from directory, because this is too big for ram
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode="binary")

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=val_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode="binary")
#take a look at one 
sample_training_images, labs = next(train_data_gen)
#Import the mobilenet model or ANY OTHER MODEL
#base_model = tf.keras.applications.ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
#base_model = tf.keras.applications.MobileNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                               include_top=False,
                                               weights='imagenet')

#########################
#We need to add a layer before the prediction layer
global_average_layer = tf.keras.layers.GlobalAveragePooling2D(
    data_format='channels_last')

#add a dense prediction layer
prediction_layer = tf.keras.layers.Dense(1)

#stack up the model
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

#set the learning rate to start
base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

#add callback to save best model based on val_acc and lower learning rate
my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',
                                        monitor='val_acc', 
                                        verbose=1,
                                        save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                         factor=0.2,
                                         patience=5)
]

#Train the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch= total_train // batch_size,
    epochs=epochs,
    callbacks=my_callbacks,
    validation_data=val_data_gen,
    validation_steps= total_val // batch_size
)

#Save metrics for later plotting
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
