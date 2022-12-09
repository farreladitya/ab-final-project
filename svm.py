import pandas
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

# load dataset
dataframe = pandas.read_csv("dataset-new4.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:1000].astype(float)
Y = dataset[:,1000]
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.20)

clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)

classifier_predictions = clf.predict(x_test)
print('Accuracy: ', accuracy_score(y_test, classifier_predictions)*100, "%")