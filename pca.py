'''
Created by xxq160330 at 5/14/2018 3:59 PM
all feature selections that have been commented out,
are the ones cannot separate clear clusters, with unclear elbow turning point, or with relatively low silhouette score
'''

#print(__doc__)
import sys, time, argparse, pathlib
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import pandas as pd
from scipy import stats

from sklearn import decomposition, preprocessing
from sklearn import datasets, metrics
from sklearn.cluster import KMeans, DBSCAN


def kmeans_distance(data, cx, cy, i_centroid, cluster_labels):
        distances = [np.sqrt((x-cx)**2+(y-cy)**2) for (x, y) in data[cluster_labels == i_centroid]]
        return distances


# kmeans
def km(n_cluster, df1, output_path, out_txtpath, v):

    #df1_std = stats.zscore(df1) # temporarily opt out, because for non-change feature, std = 0 causes null cells
    df1_std = df1
    kmeans = KMeans(n_clusters=n_cluster, verbose=v, random_state=np.random.randint(0, 1024)).fit(df1_std)
    labels = kmeans.labels_
    distances = kmeans.transform(df1_std)
    print('distances: ', distances.shape, type(distances))
    #print('labels: ', type(labels), labels)
    centroids = kmeans.cluster_centers_
    sumOfDistance = kmeans.inertia_
    with out_txtpath.open('a') as f:
        f.write('Final intertia from KMeans: ' + str(sumOfDistance) + '\n')
        f.write('Centroids: ' + str(centroids) + '\n')
        f.close()
    df1['clusters'] = labels
    #print('now with clusters, type of df1:', type(df1), 'dimension:', df1.shape)
    df['clusters'] = labels
    tempSeries = df1.groupby(by='clusters').size().sort_values()
    cluster_dict = tempSeries.to_dict()
    print('before sorting: ', cluster_dict)
    cnt = 0
    for k in cluster_dict.keys():
        cluster_dict[k] = cnt
        cnt += 1
    print('after sorting: ', cluster_dict)
    
    df['clusters'] = df['clusters'].map(cluster_dict)
    df1['clusters'] = df1['clusters'].map(cluster_dict)
    #print(df.groupby(by='clusters').groups)
    #print('now with clusters, type of df:', type(df), 'dimension:', df.shape)
    print('Final inertia after KMeans: ', sumOfDistance)
    print()
    df1_sorted = df1.sort_values(by='clusters')
    df_sorted = df.sort_values(by='clusters')
    df_sorted.to_csv(Path(output_path, str(n_cluster) + 'clusters_' + 'kmeans_label.csv'))
    
    return df1_sorted['clusters'].to_numpy(), df1_sorted

def dbscan(df1, output_path, out_txtpath, v):
    print('Running DBSCAN now...')
    df1_std = preprocessing.StandardScaler().fit_transform(df1)
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=3).fit(df1_std)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    df1['clusters'] = labels
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    #print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    #print("Adjusted Rand Index: %0.3f"
    #    % metrics.adjusted_rand_score(labels_true, labels))
    #print("Adjusted Mutual Information: %0.3f"
    #    % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(df1_std, labels))
    df['clusters'] = labels
    df.to_csv(Path(output_path, str(n_clusters_) + 'clusters_' + 'dbscan_label.csv'))
    
    coordinates, coordinates_std, df1, df1_std = pca(df1, labels, n_clusters_, out_subpath, out_txtpath)
    plotting(n_clusters_, coordinates, coordinates_std, df1, df1_std, infile, out_subpath)
    
    '''
    # #############################################################################
    # Plot result

    # Black removed and is used for noise instead.
    
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = df1_std[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = df1_std[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    '''
# PCA
def pca(df1, labels, n_cluster, output_path, out_txtpath):
    # pca
    scalar = preprocessing.StandardScaler().fit(df1)
    df1_std = scalar.transform(df1)
    pca = decomposition.PCA(n_components=2).fit(df1)
    #pca = decomposition.SparsePCA(n_components=2).fit(df1)
    pca_std = decomposition.PCA(n_components=2).fit(df1_std)
    #pca_std = decomposition.SparsePCA(n_components=2).fit(df1_std)
    #print('non-standardized variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
    print('standardized variance ratio (first two components): %s' % str(pca_std.explained_variance_ratio_))
    total_var = pca_std.explained_variance_ratio_[0] + pca_std.explained_variance_ratio_[1]
    with open(out_txtpath, 'a') as f:
        f.write('standardized variance ratio (first two components): ' + str(pca_std.explained_variance_ratio_) + '\n')
        f.write('total variance ratio (first two): ' + str(total_var) + '\n')
        f.close()

    df1 = pca.transform(df1)
    df1_std = pca_std.transform(df1_std)
    # print('type of df1, ', type(df1), df1.shape)
    (row, col) = df1_std.shape
    l = labels.T.reshape(row, 1)
    df1 = np.concatenate((df1, l), axis=1)
    coordinates = [np.where(df1[:, 2] == k) for k in range(0, n_cluster)]  # get row number for each cluster label
    df1_std = np.concatenate((df1_std, l), axis=1)
    # print('pca result dimension', df1_std.shape, 'first row', df1_std[0:2])
    coordinates_std = [np.where(df1_std[:, 2] == k) for k in range(0, n_cluster)]
    #print('index of samples in all clusters', coordinates_std)
    return coordinates, coordinates_std, df1, df1_std

# plot
def plotting(n_cluster, coordinates, coordinates_std, df1, df1_std, infile, output_path):
    # plot non-standardized and standardized together in one figure
    # plt.figure(0)
    '''
    fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
    colors = ['navy', 'tomato', 'turquoise', 'darkorange', 'red', 'orange', 'plum', 'grey', 'olive', 'brown']
    markers = ['^', 's', 'o', 'd', 'x', '1', '2', '3', '4', '*']
    
    for color, i, label in zip(colors[0:n_cluster], range(0, n_cluster), markers[0:n_cluster]):
        print('%d th cluster:' % (i + 1))
        for j in coordinates[i]:  # row number of cluster i
            ax1.scatter(df1[j, 0], df1[j, 1], color=color, alpha=.8, label='Cluster %s' % i, marker=label)

    ax1.set_title('Transformed NON-standardized Dataset after PCA')

    # plot all samples based on clusters, each cluster has its own marker
    for color, i, label in zip(colors[0:n_cluster], range(0, n_cluster), markers[0:n_cluster]):
        print('%d th cluster:' % (i + 1))
        for j in coordinates_std[i]:  # row number of cluster i
            ax2.scatter(df1_std[j, 0], df1_std[j, 1], color=color, alpha=.8, label='Cluster %s' % i, marker=label)
    ax2.set_title('Transformed Standardized Dataset after PCA')

    for ax in (ax1, ax2):
        ax.set_xlabel('$1_{st}$ Principal Component')
        ax.set_ylabel('$2_{nd}$ Principal Component')
        ax.legend(loc='upper right')
        ax.grid()
    plt.savefig('pca_normalize_or_not.eps')
    plt.show()
    '''

    # only save pca after standardization
    plt.figure()
    fig2 = plt.plot()
    colors = ['navy', 'tomato', 'turquoise', 'darkorange', 'red', 'orange', 'plum', 'grey', 'olive', 'brown']
    markers = ['^', 's', 'o', 'd', 'x', '1', '2', '3', '4', '*']
    for color, i, label in zip(colors[0:n_cluster], range(0, n_cluster), markers[0:n_cluster]):
        print('%d th cluster:' % (i + 1))
        for j in coordinates_std[i]:  # row number of cluster i
            plt.scatter(df1_std[j, 0], df1_std[j, 1], color=color, alpha=.8, label='Cluster %s' % i, marker=label)
    # plt.title('PCA of Clustered IEC104 Sessions with K = %d' % n_cluster)
    plt.xlabel('$1^{st}$ Principal Component')
    plt.ylabel('$2^{nd}$ Principal Component')
    plt.legend(loc='best')

    plt.tight_layout()
    pca_fig = str(infile) + '_' + str(n_cluster) + 'clusters_' + 'pca.eps'
    plt.savefig(Path(output_path, pca_fig))
    #plt.savefig('pca.eps')
    plt.show()

if __name__ == '__main__':
    
    # show working directory path, input path, output path
    print('Please follow this format: python pca.py @pca-args.txt; modify the arguments in this txt file')
    print('Current path: ', Path().absolute())
    input_path = Path(Path().absolute(), 'input')
    output_path = Path(Path().absolute(), 'output')
    print('INPUT path: ', input_path)
    print('OUTPUT path: ', output_path)

    # parse command-line arguments
    cl_parser = ArgumentParser(fromfile_prefix_chars='@')
    cl_parser.add_argument('--infile')
    cl_parser.add_argument('--startK')
    cl_parser.add_argument('--endK')
    cl_parser.add_argument('-v', '--verbose')
    namespace = cl_parser.parse_args()
    print(namespace)
    infile = namespace.infile
    print('input data file: ', infile)
    Ks = list(np.arange(int(namespace.startK), int(namespace.endK) + 1))
    print('testing range of cluster number k is: ', Ks)
    #cl_parser.error('Please follow this format: python findingK.py @findingK-args.txt')

    df = pd.read_csv(Path(output_path, infile), delimiter=',')
    #df = pd.read_csv(Path(input_path, 'clustering-gas', infile), delimiter=',') # gas data
    #df = pd.read_csv(Path(input_path, 'clustering-XM', infile), delimiter=',') # all XM data
    #df = pd.read_csv('../input/combTypesProcessed.csv', delimiter=',') # 2017 XM data
    #df = pd.read_csv('../output/IEC104-Analysis_part1_104only.json2020-04-09_19_35_12.csv', delimiter=',') # netherlands data
    #df = pd.read_csv('../input/IEC104-Analysis_104only_2018.json2020-04-18_00_46_36.csv', delimiter=',') # 2018 XM data

    #df1 = pd.concat([df.iloc[:, 5], df.iloc[:, 7:9]], axis=1)   # classic feature vector
    #df1 = pd.concat([df['averageTimeDelta'], df['numOfPackets'], df['apduLen']], axis=1)    # classic feature vector
    #df1 = pd.concat([df['averageTimeDelta'], df['percentS'], df['percentI'], df['percentU']], axis=1)
    #df1 = pd.concat([df['averageTimeDelta'], df['numOfPackets'], df['percentS'], df['percentI'], df['percentU'], df['Interro'],	df['Norm'],	df['SP']], axis=1)
    #df1 = pd.concat([df['numOfPackets'], df['apduLen']], axis=1)
    df1 = df.drop(['Unnamed: 0', 'RTU'], axis=1, inplace=False)
    #df1 = pd.concat([df['averageTimeDelta'], df['apduLenRate']], axis=1)
    #df1 = df.iloc[:, 7:12]
    print('current features: \n', df1.columns)

    # create new folders to store output files with current time + parameters
    cur_time = str(time.strftime("%b-%d-%Y-%H-%M-%S", time.localtime()))
    out_subpath = cur_time + '_rangeOfK=' + str(Ks).strip('[]') + '_' + str(infile[:-5])
    out_subpath = Path(output_path, out_subpath)
    print('new sub-folder for output files is created: ', out_subpath)
    out_subpath.mkdir(parents=True, exist_ok=True)
    
    for n_cluster in Ks:
        out_txtpath = Path(out_subpath, str(n_cluster) + '_clusters_pca_results.txt')
        with open(out_txtpath, 'w') as f:
            f.write(str(n_cluster) + '_clusters:\n')
            f.close()

        #df1 = pd.concat([df.iloc[:, 5], df.iloc[:, 7:9]], axis=1)   # classic feature vector
        #df1 = pd.concat([df['averageTimeDelta'], df['apduLenRate']], axis=1)
        #df1 = pd.concat([df['averageTimeDelta'], df['numOfPackets'], df['apduLen']], axis=1)    # classic feature vector
        #df1 = pd.concat([df['numS'], df['numI'], df['numU']], axis=1)
        #df1 = pd.concat([df['averageTimeDelta'], df['numOfPackets'], df['percentS'], df['percentI'], df['percentU'], df['Interro'],	df['Norm'],	df['SP']], axis=1)
        #df1 = pd.concat([df['numOfPackets'], df['apduLen']], axis=1)
        df1 = df.drop(['Unnamed: 0', 'RTU'], axis=1, inplace=False)
        print('current features: \n', df1.columns)
        labels, df1 = km(n_cluster, df1, out_subpath, out_txtpath, int(namespace.verbose))
        coordinates, coordinates_std, df1, df1_std = pca(df1, labels, n_cluster, out_subpath, out_txtpath)
        plotting(n_cluster, coordinates, coordinates_std, df1, df1_std, infile, out_subpath)
    print('kmeans++ finished!!!')
    
    #out_txtpath = Path(out_subpath, 'dbscan_clusters_pca_results.txt')
    #dbscan(df1, out_subpath, out_txtpath, 1)
    
    sys.exit(0)

# 2-d showing of PCA starting from here
#df1 = pd.concat([df.iloc[:, 7:11], df.iloc[:, 12:15]], axis=1)
#df1 = df.iloc[:, 7:12]
#df1 = pd.concat([df.iloc[:, 2], df.iloc[:, 7:12]], axis=1)
#df1 = pd.concat([df.iloc[:, 2], df.iloc[:, 5], df.iloc[:, 7:12]], axis=1)
#df = pd.read_csv('processed.csv', delimiter=',')
#df1 = df.iloc[:, 12:]
#df1 = pd.concat([df.iloc[:, 2], df.iloc[:, 5:8]], axis=1)  # select certain features
#df1 = pd.concat([df.iloc[:, 2], df.iloc[:, 5], df.iloc[:, 7:9]], axis=1)
#df1 = df.iloc[:, 7:9]
#print('initial df1 sample: \n', df1[0:5])

'''
# kmeans clustering
n_cluster = 5
df1_std = stats.zscore(df1)
kmeans = KMeans(n_clusters=n_cluster).fit(df1_std)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
df1['clusters'] = labels
print('now with clusters, type of df1:', type(df1),'dimension:',df1.shape)
df['clusters'] = labels
print('now with clusters, type of df:', type(df),'dimension:',df.shape)
df_sorted = df.sort_values(by='clusters')
df_sorted.to_csv('pca_label.csv')

# pca
scalar = preprocessing.StandardScaler().fit(df1)
df1_std = scalar.transform(df1)
pca = decomposition.PCA(n_components=2).fit(df1)
pca_std = decomposition.PCA(n_components=2).fit(df1_std)
print('non-standardized variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
print('standardized variance ratio (first two components): %s' % str(pca_std.explained_variance_ratio_))

df1 = pca.transform(df1)
df1_std = pca_std.transform(df1_std)
#print('type of df1, ', type(df1), df1.shape)
(row, col) = df1_std.shape
l = labels.T.reshape(row,1)
df1 = np.concatenate((df1,l), axis=1)
coordinates = [np.where(df1[:,2] == k) for k in range(0, n_cluster)]        # get row number for each cluster label
df1_std = np.concatenate((df1_std,l), axis=1)
#print('pca result dimension', df1_std.shape, 'first row', df1_std[0:2])
coordinates_std = [np.where(df1_std[:,2] == k) for k in range(0, n_cluster)]
print('index of samples in all clusters',coordinates_std)

# plot non-standardized and standardized together in one figure
plt.figure(1)
fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))
colors = ['navy', 'tomato', 'turquoise', 'darkorange', 'red', 'orange', 'plum', 'grey', 'olive', 'brown']
markers = ['^', 's', 'o', 'd', 'x', '1', '2', '3', '4', '*']
for color, i, label in zip(colors[0:n_cluster], range(0,n_cluster), markers[0:n_cluster]):
    print('%d th cluster:' % (i + 1))
    for j in coordinates[i]:  # row number of cluster i
        ax1.scatter(df1[j, 0], df1[j, 1], color=color, alpha=.8, label='Cluster %s' % i, marker=label)


ax1.set_title('Transformed NON-standardized Dataset after PCA')

# plot all samples based on clusters, each cluster has its own marker
for color, i, label in zip(colors[0:n_cluster], range(0,n_cluster), markers[0:n_cluster]):
    print('%d th cluster:' % (i + 1))
    for j in coordinates_std[i]:  # row number of cluster i
        ax2.scatter(df1_std[j, 0], df1_std[j, 1], color=color, alpha=.8, label='Cluster %s' % i, marker=label)
ax2.set_title('Transformed Standardized Dataset after PCA')

for ax in (ax1, ax2):

    ax.set_xlabel('$1_{st}$ Principal Component')
    ax.set_ylabel('$2_{nd}$ Principal Component')
    ax.legend(loc='upper right')
    ax.grid()
plt.savefig('pca_normalize_or_not.svg')
plt.show()

# only save pca after standardization
plt.figure(2)
fig2 = plt.plot()
colors = ['navy', 'tomato', 'turquoise', 'darkorange', 'red', 'orange', 'plum', 'grey', 'olive', 'brown']
markers = ['^', 's', 'o', 'd', 'x', '1', '2', '3', '4', '*']
for color, i, label in zip(colors[0:n_cluster], range(0,n_cluster), markers[0:n_cluster]):
    print('%d th cluster:' % (i + 1))
    for j in coordinates_std[i]:  # row number of cluster i
        plt.scatter(df1_std[j, 0], df1_std[j, 1], color=color, alpha=.8, label='Cluster %s' % i, marker=label)
#plt.title('PCA of Clustered IEC104 Sessions with K = %d' % n_cluster)
plt.xlabel('$1^{st}$ Principal Component')
plt.ylabel('$2^{nd}$ Principal Component')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('pca.svg')
plt.show()

'''