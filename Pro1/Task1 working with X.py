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
from sklearn import neighbors
 
X= np.load('extractedtrainimages.npy')
#Y_test = np.load('extractedtestimages.npy')

#specify the training images directory location
traindir = '.' + os.sep + 'train' + os.sep + 'train' + os.sep
class_names = sorted(os.listdir(traindir))
y = []
class_nameS = []
#since we have the X already, just get the y
for root, dirs, files in os.walk(traindir):
    print(root)
    for name in files:
        if name.endswith(".jpg"):
             
            # Extract class name from the directory name:
            label = root.split(os.sep)[-1]
            y.append(class_names.index(label))
 
y_train = np.array(y)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.20, random_state=50)


X_train=X_train[1:21]
y_train=y_train[1:21]

X_test=X_test[1:6]
y_test=y_test[1:6]

ytestlabels=[]
for i in range(len(y_test)):
    print(y_test[i])
    Wlabel=class_names[y_test[i]]
    print(Wlabel)
    ytestlabels.append(Wlabel)
labels=ytestlabels
    
    

    
    
# 5. Try different models
# 5. Try different models
# 5. Try different models
# 5. Try different models
# 5. Try different models
# 5. Try different models

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
#model_1 = KNeighborsClassifier (n_neighbors=10)
#model_1.fit (X_train, y_train)
#y_pred_1=model_1.predict(X_test)
#acc_1=accuracy_score(y_test,y_pred_1)
#print (acc_1)


#model_MLP = MLPClassifier ()
#model_MLP.fit (X_train, y_train)
#y_pred_1=model_MLP.predict(X_test)
#acc_1=accuracy_score(y_test,y_pred_1)
#print (acc_1)
# relu result is 0.7257257257257257
# logistic is 0.7357357357357357 so 77.05
# identity result is 0.7227227227227228
# tanh is 0.7317317317317318


#model_LDA = LinearDiscriminantAnalysis()
#model_LDA.fit (X_train, y_train)
#y_pred_1=model_LDA.predict(X_test)
#acc_1=accuracy_score(y_test,y_pred_1)
#print (acc_1)
# relu result is 0.7257257257257257
# logistic is 0.7357357357357357 so 77.05
# identity result is 0.7227227227227228
# tanh is 0.7317317317317318
#
#model_3 = SVC (gamma='sigmoid')
#model_3.fit (X_train, y_train)
#y_pred_3=model_3.predict(X_test)
#acc_3=accuracy_score(y_test,y_pred_3)
#print (acc_3)
