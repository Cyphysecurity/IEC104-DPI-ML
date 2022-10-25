'''
Created by grace at 5/13/18 12:33 PM
This script is clustering algorithm
'''

import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def second(df):
    deep_cluster = [1, 0]           # target cluster labels to further clustering
    condition = df.clusters.isin(deep_cluster)
    result = df[condition]          # dataframes filtered by target cluster label

if __name__ == '__main__':
    df = pd.read_csv('pca_label.csv', delimiter=',')
    dft = pd.concat([df.iloc[:, 0:15], df.iloc[:, 17]], axis=1)
    headers = ['item','system','asdu_type','min','max']
    second(dft)
