'''
Created by xxq160330 at 5/24/2018 6:56 PM
This script uses forward stepwise regression to select proper features to build a feature vector
Statistic metric used to measure regression is Silhouette
'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import stats

from sklearn import decomposition, preprocessing
from sklearn import datasets
from sklearn.cluster import KMeans

def km(k, X):
    n_cluster = k
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    X_transformed = kmeans.transform(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

def forward(df):
    # Step 1: start empty vector
    sf = []
    silScore = []
    Ks = range(2, 10)
    # Step 2: 1st iteration, kmeans on all individual features and measure silhouette
    features = list(df.columns)             # start with all current features
    for i in range(0, len(features)):
        cur_f = features[i]



    # Step 3: add the features with 1 highest silhouette score to sf
    # when add, print the silhouette score and add score to silScore

    # Step 4: kmeans on current sf and the rest features individually and measure silhouette for all

    # Step 5: repeat step 3

    # Step 6: stop if all individual feature silhouette point is lower than 0.5

    return (sf, silScore)

if __name__ == '__main__':
    df = pd.read_csv('combinedProcessed.csv', delimiter=',')
    df = pd.concat([df.iloc[:,2], df.iloc[:, 5:]], axis=1)
    (sf, sil) = forward(df)
    print('after selection, (feature, silhouette score) pair is: \n')
    print( '%s: %.5f ' % (str(sf[i]), sil[i]) for i in range(0, len(sf)))



