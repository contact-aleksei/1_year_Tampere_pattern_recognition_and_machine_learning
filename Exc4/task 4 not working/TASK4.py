# -*- coding: utf-8 -*-
"""
@author: OWNER
Task 3
"""
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from simplelbp import local_binary_pattern

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
N = 32 # Number of feature maps
w, h = 5, 5 # Conv. window size
model = Sequential()
model.add(Conv2D(N, (w, h), input_shape=(64, 64, 3),
                 activation = 'relu', padding = 'same'))
# max_pooling2d_47 (MaxPooling (None, 16, 16, 32) 0
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(N, (w, h), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(4, 4)))   
model.add(Flatten())
model.add(Dense(100 , activation = 'sigmoid'))
model.add(Dense(2, activation = 'sigmoid'))
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
def extract_lbp_features(X, P = 8, R = 5):
    """
    Extract LBP features from all input samples.
    - R is radius parameter
    - P is the number of angles for LBP
    """
    
    F = [] # Features are stored here
    
    N = X.shape[0]
    for k in range(N):
        
        print("Processing image {}/{}".format(k+1, N))
        
        image = X[k, ...]
        lbp = local_binary_pattern(image, P, R)
        hist = np.histogram(lbp, bins=range(257))[0]
        F.append(hist)

    return np.array(F)


X, y = load_data(".")
X, y = load_data(".")
F = extract_lbp_features(X)
print("X shape: " + str(X.shape))
print("F shape: " + str(F.shape))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,  y_test = train_test_split(F, y, test_size=0.20, random_state=50) 

# Compile and train
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test))
