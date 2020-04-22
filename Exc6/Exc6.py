from sklearn.ensemble import RandomForestClassifier
import scipy
import numpy
import matplotlib.pyplot as plot

mat_contents= scipy.io.loadmat('arcene.mat')
X_test=mat_contents['X_test']
X_train=mat_contents['X_train']
y_train=mat_contents['y_train'].ravel()
y_test=mat_contents['y_test'].ravel()

#TASK_1#TASK_1#TASK_1#TASK_1#TASK_1#TASK_1#TASK_1#TASK_1#TASK_1#TASK_1#TASK_1#

#clf = RandomForestClassifier(n_estimators =100, max_depth=2, random_state=0)
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
#from sklearn.metrics import accuracy_score
#RandomForestClassifier = accuracy_score(y_test, y_pred)
#importances = clf.feature_importances_
#plot.bar(numpy.arange(len(importances)), importances)

#TASK_2#TASK_2#TASK_2#TASK_2#TASK_2#TASK_2#TASK_2#TASK_2#TASK_2#TASK_2#TASK_2#

#from sklearn.feature_selection import RFECV
#from sklearn.linear_model import LogisticRegression
#estimator = LogisticRegression(random_state=0)
#RFE = RFECV(estimator, step=50,verbose = 1)
#RFE.fit(X_train, y_train)
## c) Count the number of selected features from rfe.support_.
#RFE.support_
## d) Plot the errors for different number of features:
#plot.plot(range(0,10001,50), RFE.grid_scores_)
#print(RFE.n_features_)

#TASK_3#TASK_3#TASK_3#TASK_3#TASK_3#TASK_3#TASK_3#TASK_3#TASK_3#TASK_3#TASK_3#

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# a) Instantiate a LogisticRegression classifier. Set penalty = ’l1’ in the
# constructor
LR = LogisticRegression(random_state=0, penalty = 'l1')

parameters = { 'C': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}
clf = GridSearchCV(LR, parameters,cv = 10)
clf.fit(X_train, y_train)

best = clf.best_params_
C = best['C']

LR = LogisticRegression(penalty = 'l1', C = C)
LR.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
sccuracy_score = accuracy_score(y_test, y_pred)

print('Accuracy X_test, y_pred :',sccuracy_score)