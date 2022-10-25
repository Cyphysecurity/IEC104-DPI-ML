'''
Created by xxq160330 at 9/17/2018 4:52 PM
This script:
1. Split dataset into training and testing
2. Benchmark learning models
3. Model evaluations
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report

# Simple split training and testing based on percentage
def simpleSplit(x, y, percent, seed):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percent, random_state=seed)
    print('train shape: x = ', x_train.shape, 'y = ', y_train.shape, 'test shape: x = ', x_test.shape, 'y = ', y_test.shape)
    return x_train, x_test, y_train, y_test

# K-Folds Cross Validation
def kfolds(X):
    kf = KFold(n_splits=33, shuffle=True, random_state=10)
    return kf.split(X)

# Leave-one-out
def loocv(X):
    return LeaveOneOut().split(X)

# KMeans
def km(n_cluster, df):
    sig = df.iloc[:, 0]
    df_std = preprocessing.RobustScaler().fit_transform(df.iloc[:, 1:])
    kmeans = KMeans(n_clusters=n_cluster).fit(df_std)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    df['clusters'] = labels
    df_sorted = df.sort_values(by='clusters')
    df_sorted.to_csv('rtu_pca_label.csv')

# SVM
def svmSvc(x_train, y_train, x_test, y_test):
    print('********** SVM SVC launched... *****************')
    clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    print('SVM SVC accuracy = %f' % accuracy)
    return accuracy

# Decision Tree
def dtree(x_train, y_train, x_test, y_test):
    print('********** Decision Tree Classifier launched... *****************')
    clf = DecisionTreeClassifier(random_state=10).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = clf.score(x_test, y_test)
    print('Decision Tree accuracy = %f' % accuracy)
    '''
        # print predicted labels and theoretical labels in two columns
        # printed result may be wrong, need debug
        compareLabel = np.concatenate((y_pred.reshape(1, y_pred.size), y_test.reshape(1, y_test.size)), axis=0).T
        print('Predicted, Actual')
        print(compareLabel)
    '''
    return accuracy, y_pred

# Naive Bayes
def gnb(x_train, y_train, x_test, y_test):
    print('********** Gaussian Naive Bayes launched... *****************')
    clf = BernoulliNB().fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    print('GNB accuracy = %f' % accuracy)
    return accuracy


# kNN
def knn(x_train, y_train, x_test, y_test):
    print('********** kNN launched... *****************')
    for i in np.arange(1, 20):
        clf = neighbors.KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
        print('i = %d, accuracy = %f' % (i, clf.score(x_test, y_test)))

# Evaluation in confusion matrix
def confusionM(y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
    #labels = ['P', 'Q', 'U', 'I', 'Status', 'Frequ', 'AGC-SP (Set-Point)', 'I-A', 'I-B', 'I-C', 'TapPosMv']
    labels = ['Power', 'Q', 'U', 'I', 'Status', 'Frequ', 'AGC-SP (Set-Point)', 'I-three', 'TapPosMv']
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, cmap='Greys', annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    plt.tight_layout()
    plt.show()

# Classification report (precision & recall) for each class
def clfreport(y_test, y_pred):
    labels = ['Power', 'Q', 'U', 'I', 'Status', 'Frequ', 'AGC-SP (Set-Point)', 'I-three', 'TapPosMv']
    print('F1 score:', f1_score(y_test, y_pred, average='macro'))
    print('Recall:', recall_score(y_test, y_pred, average='macro'))
    print('Precision:', precision_score(y_test, y_pred, average='macro'))
    print('\n clasification report:\n', classification_report(y_test, y_pred, labels=labels))
    print('\n confussion matrix:\n', confusion_matrix(y_test, y_pred))

