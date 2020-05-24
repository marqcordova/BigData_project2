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

import multiprocessing
from joblib import Parallel, delayed
import time


path_to_project = os.path.join('C:\\Users\\cordovam\\Desktop\\project 2')
# path_to_project = os.path.join('C:\\Users\\brand\\Documents\\Courses\\ST 538\\PROJECT 02\\DATA')
loaded_model = tf.keras.models.load_model(os.path.join(path_to_project, 'model.50-0.97.h5'))

# Check its architecture
loaded_model.summary()

# Get CSV filenames
test = pd.read_csv(os.path.join(path_to_project, 'test.csv'))

# Number of parallel threads to use in running the model
# multiprocessing.cpu_count() # get number of available threads
NUM_THREADS = 2

# final_pred = []
# for i in range(len(test.filename)):

# Function: run_model()
# Log predictions from model on test data
# Takes tuple(2) as argument from parallelizing calls, where first argument
# is filename, and second argument is index of filename
def run_model(fname_and_index):
    fname = fname_and_index[0]
    i = fname_and_index[1]
    # print(i)
    image_path = os.path.join(path_to_project,
                              'project_test',
                              fname)
    
    image = Image.open(image_path)
    data = np.asarray(image)/255
    data = np.expand_dims(data, axis=0)
    pred = loaded_model.predict(data)
    
    if np.exp(pred) >= 1:
        final_pred = 1
        # final_pred.append(1)
    else:
        final_pred = 0
        # final_pred.append(0)
    if i % 1000 == 0:
        print(i, 'iterations')
    
    return final_pred



tic = time.time()

# Parallelize calls to run_model(). Replaces for-loop that originally
# processed predictions.
if __name__ == "__main__":
    final_pred = Parallel(n_jobs=NUM_THREADS, backend='threading')(map(delayed(run_model), zip(test.filename, range(len(test.filename)))))

toc = time.time()


test.label[0:len(final_pred)]

correct = 0 
for i in zip(final_pred, test.label[0:len(final_pred)]):
    if i[0] == i[1]:
        # print('correct')
        correct += 1
    # else:
        # print('incorrect')
  
#Test accuracy
print('Testing time:', round(toc-tic, 3), 'm')
print('Total predictions:', len(final_pred))
print('Correct:', correct)
print('Incorrect:', len(final_pred) - correct)
print('Proportion correct:', correct/len(final_pred))