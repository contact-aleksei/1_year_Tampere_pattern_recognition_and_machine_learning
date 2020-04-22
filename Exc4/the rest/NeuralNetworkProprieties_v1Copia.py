import numpy as np
import tensorflow as tf
import os
import PIL
import keras
import platform
import matplotlib.pyplot as plt
import subprocess as sp
import time
import sys
import json
import time
import h5py
import keras
import os.path

from LoadImagesCubePlus_v1 import *
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout, Conv2D, InputLayer, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import Model, layers
from keras.models import load_model, model_from_json
from keras.optimizers import SGD
from datetime import datetime

try:
	print('\n\n\n\n')
	# create the sequential network
	model = Sequential()
	N = 10 # Number of feature maps
	w, h = 5, 5 # Conv. window size
	model.add(Conv2D(N, (w, h),\
	input_shape=(64, 64, 3),\
	activation = 'relu',\
	padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(N, (w, h),\
	activation = 'relu',\
	padding = 'same'))
	model.add(MaxPooling2D((2,2)))
	model.add(Flatten())
	model.add(Dense(2, activation = 'sigmoid'))
	model.summary()

except Exception as e:
	print(str(datetime.now()) + " Color Constancy Network: Error! \n")
	exc_type, exc_obj, exc_tb = sys.exc_info()
	fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
	print("Type error: " + str(e))
	print('File Name:', fname, 'Line Number:', exc_tb.tb_lineno, end = '\n\n')
