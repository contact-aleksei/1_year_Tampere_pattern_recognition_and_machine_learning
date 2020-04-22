import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from simplelbp import local_binary_pattern
import os
import numpy
import glob
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
    
model = Sequential()
N = 32 # Number of feature maps
w, h = 5, 5 # Conv. window size
model.add(Conv2D(N, (w, h),\
input_shape=(64, 64, 3),\
activation = 'relu',\
padding = 'same'))


model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(N, (w, h),\
activation = 'relu',\
padding = 'same'))
model.add(MaxPooling2D((4,4)))
model.add(Flatten())
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dense(2, activation = 'sigmoid'))
model.summary()

# =============================================================================

SCG = keras.optimizers.SGD(lr=0.01, momentum=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=SCG, metrics=['accuracy'])

# =============================================================================

""" 
Load all images from subdirectories of
'folder'. The subdirectory name indicates
the class.
"""

X = []          # Images go here
y = []          # Class labels go here
classes = []    # All class names go here
folder = os.getcwd()
subdirectories = glob.glob(folder + "/*")

index=0
for d in subdirectories:

    files = glob.glob(d + os.sep + "*.jpg")
    for name in files:
        index +=1
        
X=numpy.empty([index, 64, 64, 3])

index=0

# Loop over all folders
for d in subdirectories:
    
    # Find all files from this folder
    files = glob.glob(d + os.sep + "*.jpg")
    
    # Load all files
    for name in files:
        
        # Load image and parse class name
        
        img = image.load_img(name, target_size=(64, 64))
        class_name = name.split(os.sep)[-2]

        # Convert class names to integer indices:
        if class_name not in classes:
            classes.append(class_name)
        
        class_idx = classes.index(class_name)
        
        X[index,:,:,:]=img
        index +=1
        y.append(class_idx)

# Convert python lists to contiguous numpy arrays
y = numpy.array(y)
y = keras.utils.to_categorical (y, 2)


print(y.shape)
print(X.shape)

Xindex = round(X.shape[0]*0.8)
yindex = round(y.shape[0]*0.8)
X_train=X[:Xindex,:,:]
y_train=y[:yindex,]
X_test=X[:-(X.shape[0]-Xindex),:,:]
y_test=y[:-(y.shape[0]-yindex),]


batch_size= 32
epochs=20

model.fit(X_train, y_train, validation_data = ([X_test, y_test]),batch_size=batch_size, epochs=epochs)