#load temsorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

#########Hyper Parameters############
#Feed 256 images into the model at a time 
batch_size = 256
#40 epochs of training should be enough to fit the model(s)
epochs = 40
#Image dimensions
IMG_HEIGHT = 96
IMG_WIDTH = 96

#Image generators. Later we will add data augmentation to the training image generator ONLY
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data

validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

#We're going to use flow from directory, because this is too big for ram
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=val_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


#Take a look at some images
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20,20))
#     axes = axes.flatten()
#     for img, ax in zip( images_arr, axes ):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# plotImages(sample_training_images[:5])

#A basic starter model with only conv and pool layers
#basic model is using 3x3 kernels at 16,32 and 64 per conv layer
#############PICK A NETWORK
# simple network
# model = Sequential([
#     Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
#     MaxPooling2D(),
#     Conv2D(32, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Conv2D(64, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(1)
# ])

# simple network with dropout
#This area is good to play around with other models
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
            input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
  #If we want a deeper model uncomment these 2 layers
    #Conv2D(128, 3, padding='same', activation='relu'),
    #MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

#look at the architecture of the model before fitting
model.summary()

#fit the model and save results as 'history'
history = model.fit_generator(
    train_data_gen,
  #This many steps means each training image is used ~1x per epoch
    steps_per_epoch= total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

#extract the accuracy and loss values for each epoch of on train and val
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

#Save the training performance data so that we can make plots in R later
filename = 'dropoutF_augF_simplest_mod.csv'
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
