'''
Created by xxq160330 at 5/15/2018 5:48 PM
This script only runs Kmeans
'''

from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

#df = pd.read_csv('../input/combinedProcessed.csv', delimiter=',')
df = pd.read_csv('../output/IEC104-Analysis_part1_104only.json2020-04-09_19_35_12.csv', delimiter=',')
df1 = df.iloc[:, 7:9]
dft = stats.zscore(df1)
#result = open('Centroids_numIoa_and_asduAddrIoa.txt', 'w')

# plot 2d Kmeans results when only two features
kmeans = KMeans(n_clusters=3)
kmeans.fit(dft)
labels = kmeans.predict(dft)
centroids = kmeans.cluster_centers_
print('centroids: ', centroids)
print('kmeans finished')

fig = plt.figure(figsize=(10, 10))
colmap = {1: 'r', 2: 'g', 3: 'b'}
colors = map(lambda x: colmap[x+1], labels)

plt.scatter(df1[df1.columns[0]], df1[df1.columns[1]])
print('all data points drawn')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlabel(df1.columns[0])
plt.ylabel(df1.columns[1])
print('centroids drawn')
plt.show()
print('image shown')
'''
#df1 = pd.concat([df.iloc[:, 2], df.iloc[:, 6:]], axis=1)  # select certain features
#dft = pd.concat([df.iloc[:, 0:3], df.iloc[:, 5:]], axis=1)  # select certain features
df1 = df.iloc[:, 12:]
print('*****************************\ncurrent feature vector is "%s" ' % list(df1.columns))
#df1 = pd.Series.to_frame(df1)
df1_std = stats.zscore(df1)

# df1_std = preprocessing.Normalizer().fit_transform(df1)   # another scalar which is robust over outliers but sensitive to negative values
n_cluster = 6
kmeans = KMeans(n_clusters=n_cluster).fit(df1)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
df1['clusters'] = labels
#print('type and shape of df1:',)
print(df1.groupby(['clusters']).mean())
print(df1.groupby(['clusters']).min())
print(df1.groupby(['clusters']).max())
result.writelines(df1.groupby(['clusters']).mean())
#print('centroids: ', centroids)
df1_l = np.concatenate((df1, labels.T.reshape(df1.shape[0], 1)), axis=1)
coordinates = [np.where(df1_l[:,2] == k) for k in range(0, n_cluster)]

plt.plot()
# prepare list of 10 colors and markers
colors = ['navy', 'tomato', 'turquoise', 'darkorange', 'red', 'yellow', 'plum', 'grey', 'olive', 'brown']
markers = ['^', 's', 'o', 'd', 'x', '1', '2', '3', '4', '*']
for color, i, label in zip(colors[0:n_cluster], range(0,n_cluster), markers[0:n_cluster]):
    print('%d th cluster:' % (i + 1))
    for j in coordinates[i]:  # row number of cluster i
        plt.scatter(df1_l[j, 0], df1_l[j, 1], color=color, alpha=.8, label='class %s' % i, marker=label)

plt.xlabel('Number of IOA')
plt.ylabel('Number of ASDU.addr+IOA')
plt.legend(loc='lower right')
plt.grid()
#plt.title('Clustering on number of IOA and ASDU_addr.ioa with k = %d' % n_cluster)
plt.savefig('kmeans.svg')
plt.show()
'''






