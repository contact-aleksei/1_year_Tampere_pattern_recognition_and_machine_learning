# -*- coding: utf-8 -*-
"""
@author: OWNER
PATTERN RECOGNITION'
"""
# %%
"FOURTH TASK"
# 4. python Implement a sinusoid detector.
# In this exercise we generate a noisy sinusoid with known frequency and see
# how the sinusoid detector of the lecture slides performs.

import matplotlib.pyplot as plt
import numpy as np

# a) Create a vector of zero and sinusoidal components that looks like the plot
# below. Commands: np.zeros, np.concatenate. Sinusoid is generated
# by np.cos(2 * np.pi * 0.1 * n).

a = np.zeros(500)
n = np.arange(100)
b = np.cos(2 * np.pi * 0.1 * n)
c = np.zeros(300)
y = np.concatenate((a, b, c))

#b) Create a noisy version of the signal by adding Gaussian noise with variance
#"0.5: y_n = y + np.sqrt(0.5) * np.random.randn(y.size).

y_n = y + np.sqrt(0.5) * np.random.randn(y.size);

# 1c) Implement the deterministic sinusoid detector (slide 20 of slideset 3)

deterministic=np.convolve(np.cos(2 *0.1* np.pi*n), y_n)
  
# d) Implement the random signal version (slide 24 of slideset 3).
    
    # x[n] exp(−2πif0n)
                            # *f0
h = np.exp(-2 * np.pi * 1j * 0.1 * n)

# z = np.abs(np.convolve(h, xn, ’same’))
z =  np.abs(np.convolve(h, y_n, 'same'))

# e) Generate plots like the ones below. Hint for plotting:
fig, ax = plt.subplots(4, 1) # Create a figure with 4 axes
ax[0].plot(y) # This will be the topmost axis
ax[1].plot(y_n) # This will be the second axis
ax[2].plot(deterministic) # This will be the 3rd axis
ax[3].plot(z) # This will be the 4th axis
plt.show() # Display on screen.

# %%
"SECOND TASK"

from scipy.io import loadmat

# a) Load the file twoClassData.mat to your python workspace.
mat = loadmat('twoClassData.mat')
# b) Split the data into training and testing sets: samples X[:200] are for training and X[200:] for testing.

y = mat["y"].ravel()
X = mat["X"]

# The function ravel() transforms y from 400 × 1 matrix into a 400-length array

from sklearn.model_selection import train_test_split
# Split arrays or matrices into random train and test subsets   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50) 
# train_size : float, int, or None, (default=None) If float
# should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split
 


# c) Train a KNN classifier. Use default parameters and compute the accuracy
# using sklearn.metrics.accuracy_score on the test set.
from sklearn.neighbors import KNeighborsClassifier

KNclassifier = KNeighborsClassifier(n_neighbors=5)
KNclassifier.fit(X_train, y_train)
y_pred = KNclassifier.predict(X_test)


from sklearn.metrics import accuracy_score
KN=accuracy_score(y_test, y_pred)



# d) Train an LDA classifier. Use default parameters and compute the accuracy
# using sklearn.metrics.accuracy_score on the test set.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

LDAclassifier = LinearDiscriminantAnalysis()
LDAclassifier.fit(X_train, y_train)
y_pred = LDAclassifier.predict(X_test)


from sklearn.metrics import accuracy_score
LDA = accuracy_score(y_test, y_pred)