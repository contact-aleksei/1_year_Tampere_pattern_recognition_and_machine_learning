# -*- coding: utf-8 -*-
"""
GSTRB folder
@author: OWNER
Task 3
"""
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
N = 32 # Number of feature maps
w, h = 5, 5 # Conv. window size
model = Sequential()
model.add(Conv2D(N, (w, h), input_shape=(64, 64, 1),
                 activation = 'relu', padding = 'same'))
# max_pooling2d_47 (MaxPooling (None, 16, 16, 32) 0
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(N, (w, h), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(4, 4)))   
model.add(Flatten())
model.add(Dense(100 , activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.summary()

def load_data(folder):
    """ 
    Load all images from subdirectories of
    'folder'. The subdirectory name indicates
    the class.
    """
    
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    
    subdirectories = glob.glob(folder + "/*")
    
    # Loop over all folders
    for d in subdirectories:
        
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
        
        # Load all files
        for name in files:
            
            # Load image and parse class name
            img = plt.imread(name)
            class_name = name.split(os.sep)[-2]

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)
            
            class_idx = classes.index(class_name)
            
            X.append(img)
            y.append(class_idx)
    
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


X, y = load_data(".")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.20, random_state=50)
X_train = X_train[..., np.newaxis] / 255.0
X_test = X_test[..., np.newaxis] / 255.0

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Output has to be one-hot-encoded
# Compile and train
model.compile(loss='categorical_crossentropy', optimizer='SGD',
              metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 20, batch_size=32, validation_data = [X_test, y_test] )
