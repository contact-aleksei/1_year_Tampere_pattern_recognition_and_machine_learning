# -*- coding: utf-8 -*-
"""
@author: OWNER
Task 3
"""
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.keys())
import matplotlib.pyplot as plt
plt.gray()
plt.imshow(digits.images[0])
plt.show()

# Split the data to training and testing sets, such that the training set
# consists of 80% and test set 20% of the data. Use
# sklearn.cross_validation.train_test_split to do this and create
# variables x_train, y_train, x_test, y_test.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(digits.data, digits.target,
                                               test_size=0.20, random_state=50)

# Create a list of four classifiers with their default parameters:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

model_1 = KNeighborsClassifier (n_neighbors=3)
model_2 = LinearDiscriminantAnalysis()
model_3 = SVC (gamma='auto')
model_4 = LogisticRegression()

# Fit the model using X as training data and y as target values
model_1.fit (X_train, y_train)
model_2.fit (X_train, y_train)
model_3.fit (X_train, y_train)
model_4.fit (X_train, y_train)

# Predict the class labels for the provided data
y_pred_1=model_1.predict(X_test)
y_pred_2=model_2.predict(X_test)
y_pred_3=model_3.predict(X_test)
y_pred_4=model_4.predict(X_test)

from sklearn.metrics import accuracy_score
# Accuracy classification score.
acc_1=accuracy_score(y_test,y_pred_1)
acc_2=accuracy_score(y_test,y_pred_2)
acc_3=accuracy_score(y_test,y_pred_3)
acc_4=accuracy_score(y_test,y_pred_4)

print (acc_1)
print (acc_2)
print (acc_3)
print (acc_4)
