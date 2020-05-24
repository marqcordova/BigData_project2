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

###############more efficient
filepaths = []
for i in test.filename:
    filepaths.append(os.path.join(path_to_project,'project_test',i))

test['filepath'] = filepaths

file_batch = 2000

final_pred= []

for i in range(20):
    test_ims = test.filepath[0+(file_batch*i):file_batch+(file_batch*i)].map(Image.open)
    
    K_test = np.stack(test_ims)/255
    
    predictions = loaded_model.predict(K_test)
    print(i)
    for j in predictions:
        if np.exp(j) >= 1:
            final_pred.append(1)
        else:
            final_pred.append(0)
            
len(final_pred)

test.label[0:len(final_pred)]

correct = 0 
for i in zip(final_pred, test.label[0:len(final_pred)]):
    if i[0] == i[1]:
       # print('correct')
        correct += 1
   # else:
        #print('incorrect')
  
#Test accuracy
print(correct/len(final_pred))