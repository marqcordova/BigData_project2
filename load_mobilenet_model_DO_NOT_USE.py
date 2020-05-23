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
from PIL import Image

path_to_project = os.path.join('C:\\Users\\cordovam\\Desktop\\project 2')
loaded_model = tf.keras.models.load_model(os.path.join(path_to_project, 'model.50-0.97.h5'))

# Check its architecture
loaded_model.summary()

test = pd.read_csv(os.path.join(path_to_project, 'test.csv'))

final_pred = []

#jack, this is ohrribly inefficient, but I am tired. Sorry :p
for i in range(len(test.filename)):
    image_path = os.path.join(path_to_project,
                              'project_test',
                              test.filename[i])
    
    image = Image.open(image_path)
    data = np.asarray(image)/255
    data = np.expand_dims(data, axis=0)
    pred = loaded_model.predict(data)
    
    if np.exp(pred) >= 1:
        final_pred.append(1)
    else:
        final_pred.append(0)
    if i % 1000 == 0:
        print(i)

len(final_pred)

test.label[0:len(final_pred)]

correct = 0 
for i in zip(final_pred, test.label[0:len(final_pred)]):
    if i[0] == i[1]:
        print('correct')
        correct += 1
    else:
        print('incorrect')
  
#Test accuracy
print(correct/len(final_pred))

