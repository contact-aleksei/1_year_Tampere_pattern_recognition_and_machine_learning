# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:48:15 2019

@author: OWNER
"""
import glob
import cv2
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from tensorflow import keras

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


class_names = sorted(os.listdir(r"C:\vehicle\train\train"))
base_model = tf.keras.applications.mobilenet.MobileNet(
        input_shape = (224,224,3),
        include_top = False)

base_model.summary() #You can get a listing of the network structure

in_tensor = base_model.inputs[0] # Grab the input of base model
out_tensor = base_model.outputs[0] # Grab the output of base model
# Add an average pooling layer (averaging each of the 1024 channels):
out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)
# Define the full model by the endpoints.
model = tf.keras.models.Model(inputs = [in_tensor],
outputs = [out_tensor])
# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model.compile(loss = "categorical_crossentropy", optimizer = 'adam')

# 4. Load all images and apply the network to each
# 4. Load all images and apply the network to each
# 4. Load all images and apply the network to each
# 4. Load all images and apply the network to each
# 4. Load all images and apply the network to each


# Find all image files in the data directory.
X = [] # Feature vectors will go here.
y = [] # Class ids will go here.
for root, dirs, files in os.walk(r"C:\vehicle\train\train"):
    for name in files:
        if name.endswith(".jpg"):
            # Load the image:
            img = plt.imread(root + os.sep + name)
            # Resize it to the net input size:
            img = cv2.resize(img, (224,224))
            # Convert the data to float, and remove mean:
            img = img.astype(np.float32)
            img -= 128
            # Push the data through the model:
            x = model.predict(img[np.newaxis, ...])[0]
             
            # And append the feature vector to our list.
            X.append(x)
            # Extract class name from the directory name:
            #label = name.split(os.sep)[-1]
            #y.append(class_names.index(label))
            
            label = root.split(os.sep)[-1]
            y.append(class_names.index(label))
            # Cast the python lists to a numpy array.
X = np.array(X)
y = np.array(y)

# 5. Try different models
# 5. Try different models
# 5. Try different models
# 5. Try different models
# 5. Try different models
# 5. Try different models
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.20, random_state=50)






