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

#Change this path to the main project directory
path_to_project = os.path.join('C:\\Users\\cordovam\\Desktop\\project 2')

#Load model weights
loaded_model = tf.keras.models.load_model(os.path.join(path_to_project, 'model.50-0.97.h5'))

# Check its architecture
loaded_model.summary()

#load the csv with the filenames and labels
test = pd.read_csv(os.path.join(path_to_project, 'test.csv'))

#Make a list of all of the test label filepaths
filepaths = []
for i in test.filename:
    filepaths.append(os.path.join(path_to_project,'project_test',i))

#add that list to the pd df
test['filepath'] = filepaths

#Load 2000 images at a time for evaluation
#This works on my 1600super, lower the number if running cpu
file_batch = 2000

final_pred= []
#use map to speed up the loading of the images
for i in range(20):
    test_ims = test.filepath[0+(file_batch*i):file_batch+(file_batch*i)].map(Image.open)
    #divide by 255 to scale all inputs from 0 to 1
    K_test = np.stack(test_ims)/255
    #make predictions on each of the 2000 images
    predictions = loaded_model.predict(K_test)
    print(i)
    #The prediction values are logits so we need to exponentiate them
    for j in predictions:
        if np.exp(j) >= 1:
            final_pred.append(1)
        else:
            final_pred.append(0)
#check to make sure we got them all            
len(final_pred)

#append predictions to dataframe
test.label[0:len(final_pred)]

#Compare the predictions to the labels and count up the correct preds
correct = 0 
for i in zip(final_pred, test.label[0:len(final_pred)]):
    if i[0] == i[1]:
       # print('correct')
        correct += 1
   # else:
        #print('incorrect')
  
#Test accuracy by dividing number correct/total number
print(correct/len(final_pred))
