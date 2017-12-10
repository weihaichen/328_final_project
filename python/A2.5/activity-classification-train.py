# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:02:58 2016

@author: cs390mb

Assignment 2 : Activity Recognition

This is the starter script used to train an activity recognition
classifier on accelerometer data.

See the assignment details for instructions. Basically you will train
a decision tree classifier and vary its parameters and evalute its
performance by computing the average accuracy, precision and recall
metrics over 10-fold cross-validation. You will then train another
classifier for comparison.

Once you get to part 4 of the assignment, where you will collect your
own data, change the filename to reference the file containing the
data you collected. Then retrain the classifier and choose the best
classifier to save to disk. This will be used in your final system.

Make sure to chek the assignment details, since the instructions here are
not complete.

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from features import extract_features # make sure features.py is in the same directory
from util import slidingWindow, reorient, reset_vars
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import pickle


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = os.path.join('data', 'activity-data.csv')
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)


# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

# you may want to play around with the window and step sizes
#default window_size = 100, step_size = 100
window_size = 20
step_size = 20

# sampling rate for the sample data should be about 25 Hz; take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

feature_names = ["mean X", "mean Y", "mean Z","Variance X","Variance Y", "Variance Z","ZCR","Magnitude-signal","Xfft","Yfft","Zfft","Entropy","Acceleration"]

class_names = ["Jogging", "Jumping","Walking","Sitting","Falling","Sitting Down","Squating","Dropping Phone"]

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

n_features = len(feature_names)

X = np.zeros((0,n_features))
y = np.zeros(0,)

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    # omit timestamp and label from accelerometer window for feature extraction:
    window = window_with_timestamp_and_label[:,1:-1]
    # extract features over window:
    x = extract_features(window)
    # append features:
    X = np.append(X, np.reshape(x, (1,-1)), axis=0)
    # append label:
    y = np.append(y, window_with_timestamp_and_label[10, -1])

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Plot data points
#
# -----------------------------------------------------------------------------

# We provided you with an example of plotting two features.
# We plotted the mean X acceleration against the mean Y acceleration.
# It should be clear from the plot that these two features are alone very uninformative.
"""print("Plotting data points...")
sys.stdout.flush()
plt.figure()
formats = ['bo', 'go']
for i in range(0,len(y),10): # only plot 1/10th of the points, it's a lot of data!
    plt.plot(X[i,0], X[i,1], formats[int(y[i])])

plt.show()
"""
# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

n = len(y)
n_classes = len(class_names)

print("\n")
print("---------------------- Decision Tree -------------------------")

trees = [] # various decision tree classifiers
trees.append(DecisionTreeClassifier(criterion="entropy", max_depth=9))
trees.append(DecisionTreeClassifier(criterion="gini", max_depth=10))
trees.append(DecisionTreeClassifier(criterion="entropy", max_depth=11))
trees.append(DecisionTreeClassifier(criterion="entropy", max_depth=12))

for tree_index, tree in enumerate(trees):

   total_accuracy = 0.0
   total_precision = [0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0]
   total_recall = [0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0]

   cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)
   for i, (train_indexes, test_indexes) in enumerate(cv):
       X_train = X[train_indexes, :]
       y_train = y[train_indexes]
       X_test = X[test_indexes, :]
       y_test = y[test_indexes]

       print("Fold {} : Training decision tree classifier over {} points...".format(i, len(y_train)))
       sys.stdout.flush()
       tree.fit(X_train, y_train)
       print("Evaluating classifier over {} points...".format(len(y_test)))

       # predict the labels on the test data
       y_pred = tree.predict(X_test)

       # show the comparison between the predicted and ground-truth labels
       conf = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7])
       print conf
       accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
       precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
       recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

       total_accuracy += accuracy
       total_precision += precision
       total_recall += recall

       print("The accuracy is {}".format(accuracy))
       print("The precision is {}".format(precision))
       print("The recall is {}".format(recall))

       print("\n")
       sys.stdout.flush()

   print("The average accuracy is {}".format(total_accuracy/10.0))
   print("The average precision is {}".format(total_precision/10.0))
   print("The average recall is {}".format(total_recall/10.0))

   print("Training decision tree classifier on entire dataset...")

# TODO: Evaluate another classifier, i.e. SVM, Logistic Regression, k-NN, etc.
#######################################################################################################################################################
# TODO: Once you have collected data, train your best model on the entire
# dataset. Then save it to disk as follows:

print("\n")
print("---------------------- Support Vector Classifier -------------------------")
svc=svm.SVC()
total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0]
total_recall = [0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0]

cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)
for i, (train_indexes, test_indexes) in enumerate(cv):
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]

    print("Fold {} : Training Support Vector classifier over {} points...".format(i, len(y_train)))
    sys.stdout.flush()
    svc.fit(X_train, y_train)
    print("Evaluating classifier over {} points...".format(len(y_test)))

    # predict the labels on the test data
    y_pred = svc.predict(X_test)

    # show the comparison between the predicted and ground-truth labels
    conf = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7])
    print conf
    accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
    precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
    recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall

    print("The accuracy is {}".format(accuracy))
    print("The precision is {}".format(precision))
    print("The recall is {}".format(recall))

    print("\n")
    sys.stdout.flush()

print("The average accuracy is {}".format(total_accuracy/10.0))
print("The average precision is {}".format(total_precision/10.0))
print("The average recall is {}".format(total_recall/10.0))

print("Training decision tree classifier on entire dataset...")


print("\n")
print("---------------------- RandomForestClassifier -------------------------")
# TODO: Train and evaluate your decision tree classifier over 10-fold CV.
# Report average accuracy, precision and recall metrics.
total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0]
total_recall = [0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0]
clf = RandomForestClassifier(n_estimators=100)
cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)
for i, (train_indexes, test_indexes) in enumerate(cv):
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]

    print("Fold {} : Training RandomForest classifier over {} points...".format(i, len(y_train)))
    sys.stdout.flush()
    clf.fit(X_train, y_train)
    print("Evaluating classifier over {} points...".format(len(y_test)))

    # predict the labels on the test data
    y_pred = clf.predict(X_test)

    # show the comparison between the predicted and ground-truth labels
    conf = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7])
    print conf
    accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
    precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
    recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall

    print("The accuracy is {}".format(accuracy))
    print("The precision is {}".format(precision))
    print("The recall is {}".format(recall))

    print("\n")
    sys.stdout.flush()
clf.fit(X,y)
print("The average accuracy is {}".format(total_accuracy/10.0))
print("The average precision is {}".format(total_precision/10.0))
print("The average recall is {}".format(total_recall/10.0))

print("\n")
print("---------------------- LogisticRegression -------------------------")
# TODO: Train and evaluate your decision tree classifier over 10-fold CV.
# Report average accuracy, precision and recall metrics.
total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0]
total_recall = [0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0]
lr = LogisticRegression()
cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)
for i, (train_indexes, test_indexes) in enumerate(cv):
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]

    print("Fold {} : Training LogisticRegression classifier over {} points...".format(i, len(y_train)))
    sys.stdout.flush()
    lr.fit(X_train, y_train)
    print("Evaluating classifier over {} points...".format(len(y_test)))

    # predict the labels on the test data
    y_pred = lr.predict(X_test)

    # show the comparison between the predicted and ground-truth labels
    conf = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7])
    print conf
    accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
    precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
    recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall

    print("The accuracy is {}".format(accuracy))
    print("The precision is {}".format(precision))
    print("The recall is {}".format(recall))

    print("\n")
    sys.stdout.flush()

print("The average accuracy is {}".format(total_accuracy/10.0))
print("The average precision is {}".format(total_precision/10.0))
print("The average recall is {}".format(total_recall/10.0))

"""svc=svm.LinearSVC()
cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)

total_fold_matrix = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

precision_total_count = np.array([0.,0.,0.,0.])
precision_total = np.array([0.,0.,0.,0.])

recall_total_count = np.array([0.,0.,0.,0.])
recall_total = np.array([0.,0.,0.,0.])

for i, (train_indexes, test_indexes) in enumerate(cv):
    print ("Fold {} : The confusion matrix is :".format(i))
    x_train = X[train_indexes, :]
    y_train = y[train_indexes]
    x_test = X[test_indexes, :]
    y_test = y[test_indexes]
    svc.fit(x_train, y_train)
    y_pred=svc.predict(x_test)

    conf = confusion_matrix(y_test, y_pred,labels=[0,1,2,3])
    print conf
    correctness = 0
    total = 0
    for x_coord in range(0, 4):
        for y_coord in range(0,4):
            if(x_coord == y_coord):
                correctness+=conf[x_coord,y_coord]
            total+=conf[x_coord,y_coord]
            total_fold_matrix[x_coord,y_coord]+=conf[x_coord,y_coord]

    print ("Accuracy:{}".format(float(correctness)/float(total)))

    # Printing precision, precision = true positive / (true positive + false positive)

    correctness = 0
    total = 0
    local_correctness = 0
    local_total = 0
    local_precision = 0
    for x_coord in range(0,4):
        for y_coord in range(0,4):
            if(x_coord == y_coord):
                for i in range(0,4):
                    total += conf[i, y_coord]
                    local_total += conf[i, y_coord]
                local_correctness = conf[x_coord, y_coord]
                correctness += conf[x_coord, y_coord]
                #print(local_total)
                if(local_total != 0):
                    precision_total[x_coord] += float(local_correctness)/float(local_total)
                    local_precision += float(local_correctness)/float(local_total)
                precision_total_count += 1
                local_correctness = 0
                local_total = 0


    print ("Precision:{}".format(float(local_precision/4)))

    # Printing Recall, recall = true positive / (true positive + false negative)

    correctness = 0
    total = 0
    local_correctness = 0
    local_total = 0
    local_recall = 0
    for x_coord in range(0,4):
        for y_coord in range(0,4):
            if(x_coord == y_coord):
                for i in range(0,4):
                    total += conf[x_coord, i]
                    local_total += conf[x_coord, i]
                local_correctness = conf[x_coord, y_coord]
                correctness += conf[x_coord, y_coord]
                if(local_total != 0):
                    recall_total[x_coord] += float(local_correctness)/float(local_total)
                    local_recall += float(local_correctness)/float(local_total)
                recall_total_count += 1
                local_correctness = 0
                local_total = 0

    print ("Recall:{}".format(float(local_recall/4)))


    print("\n")

correctness = 0
total = 0
for x_coord in range(0,4):
    for y_coord in range(0,4):
        if x_coord == y_coord:
            correctness += total_fold_matrix[x_coord, y_coord]
        total += total_fold_matrix[x_coord, y_coord]
print ("All folds average accuracy: {}".format((float(correctness)/float(total))))

# Computing all folds average precision

print ("All folds average precision, Jogging: {}".format(float(precision_total[0]/10)))
print ("All folds average precision, Jumping: {}".format(float(precision_total[1]/10)))
print ("All folds average precision, Walking: {}".format(float(precision_total[2]/10)))
print ("All folds average precision, Sitting: {}".format(float(precision_total[3]/10)))

print ("All folds average recall, Jogging: {}".format(float(recall_total[0]/10)))
print ("All folds average recall, Jumping: {}".format(float(recall_total[1]/10)))
print ("All folds average recall, Walking: {}".format(float(recall_total[2]/10)))
print ("All folds average recall, Sitting: {}".format(float(recall_total[3]/10)))"""

# when ready, set this to the best model you found, trained on all the data:
best_classifier = clf

with open('classifier.pickle', 'wb') as f: # 'wb' stands for 'write bytes'
    pickle.dump(best_classifier, f)
