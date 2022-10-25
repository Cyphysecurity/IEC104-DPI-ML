'''
Created by xxq160330 at 5/22/2018 5:48 PM
This script is to find the best K for Kmeans algorithm
1. elbow method
2. silhouette score/coefficient
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import stats
from scipy.spatial.distance import cdist
import sys, argparse, time, pathlib
from pathlib import Path
from argparse import ArgumentParser 

def inertia(km, df):
    # following lists all indexed by k
    kmeans = [km0.fit(df) for km0 in km]
    centroids = [m.cluster_centers_ for m in kmeans]
    inertias = [sum(np.min(cdist(df, center, 'euclidean'), axis=1)) for center in centroids]
    return (inertias, True)

def elbow(km, df):
    # following lists all indexed by k
    kmeans = [km0.fit(df) for km0 in km]
    centroids = [m.cluster_centers_ for m in kmeans]
    distortions = [sum(np.min(cdist(df, center, 'euclidean'), axis=1)) / df.shape[0] for center in centroids]
    return (distortions, True)

def silhouette(Ks, km, df, outpath):
    silfile = open(Path(outpath, 'sil_avg.txt'), 'w')  # save the average silhouette score for all clusters
    # following lists all indexed by k
    kmeans = [km0.fit(df) for km0 in km]
    labels = [m.labels_ for m in kmeans]
    #print('type of labels', type(labels))
    #print('shape of labels', len(labels))
    #print('each array size', labels[0].shape)
    #print('3rd array: ', labels[3])
    sil_avg = [silhouette_score(df, label) for label in labels]     # average silhouette score for all samples
    for k in range(len(kmeans)):
        line = 'For n_cluster = %d, average silhouette score = %.5f\n' % (k+2, sil_avg[k])
        silfile.write(line)
        print(line)
    sil_each = [silhouette_samples(df, label) for label in labels]  # sil score for every sample
    #print('type and shape of sil_each', type(sil_each[1]), sil_each[1].shape)

    '''
    # plot silhouette 
    for n_cluster in Ks:
        k = Ks.index(n_cluster)
        #k, = np.where(Ks == n_cluster)
        # for each k, plot individual figure for silhouette value distribution over all samples
        fig = plt.plot()
        plt.xlim([-1,1])
        plt.ylim([0, len(df) + (n_cluster + 1) * 10])                       # (n_cluster+1) * 10 is the blank between clusters

        low_y = 0
        print('%d clusters' % n_cluster)
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_each[k][labels[k] == i]
            #print('type and shape of %d th_cluster_sil_values' % i, type(ith_cluster_sil_values), ith_cluster_sil_values.shape)
            ith_cluster_sil_values.sort()
            size_cluster_i = ith_cluster_sil_values.shape[0]
            high_y = low_y + size_cluster_i
            color = cm.Spectral(float(i) / n_cluster)                       # spread color for each cluster
            plt.fill_betweenx(np.arange(low_y, high_y), 0, ith_cluster_sil_values, facecolor=color, edgecolor=color)
            plt.text(-0.05, low_y + 0.5 * size_cluster_i, str(i))   # text label cluster number in the middle of each cluster
            low_y = high_y + 10
        plt.title('Silhouette scores for clusters with %d clusters' % n_cluster)
        plt.xlabel('Silhouette coefficient values')
        plt.ylabel('Cluster labels')
        plt.axvline(x=sil_avg[k], color='red', linestyle='--')         # add vertical line to show average sil score
        plt.savefig(Path(outpath, 'combined_silhouette_%d clusters.eps' % n_cluster))
        #plt.show()
    '''
if __name__ == '__main__':
    # show working directory path, input path, output path
    print('Please follow this format: python findingK.py @findingK-args.txt; modify the arguments in this txt file')
    print('Current path: ', Path().absolute())
    input_path = Path(Path().parent.parent.absolute(), 'input')
    output_path = Path(Path().parent.parent.absolute(), 'output')
    print('INPUT path: ', input_path)
    print('OUTPUT path: ', output_path)

    # parse command-line arguments
    cl_parser = ArgumentParser(fromfile_prefix_chars='@')
    cl_parser.add_argument('--infile')
    cl_parser.add_argument('-startK')
    cl_parser.add_argument('-endK')
    namespace = cl_parser.parse_args()
    print(namespace)
    infile = namespace.infile
    print('input data file: ', infile)
    Ks = list(np.arange(int(namespace.startK), int(namespace.endK) + 1))
    print('testing range of cluster number k is: ', Ks)
    #cl_parser.error('Please follow this format: python findingK.py @findingK-args.txt')


    # getting data
    #df = pd.read_csv('../output/IEC104-Analysis_part1_104only.json2020-04-09_19_35_12.csv', delimiter=',') # Netherland, tried feature vector: [9:12]
    #df = pd.read_csv('../input/IEC104-Analysis_104only_2018.json2020-04-18_00_46_36.csv', delimiter=',') # 2018 XM data
    #df = pd.read_csv(Path(input_path, 'combTypesProcessed.csv'), delimiter=',') # 2017 XM data
    #df = pd.read_csv(Path(input_path, 'clustering-XM', infile), delimiter=',') # 2017 XM data
    df = pd.read_csv(Path(input_path, 'clustering-gas', infile), delimiter=',') # gas part 31 data
    print(df.columns)
    #dft = pd.concat([df['averageTimeDelta'], df['numOfPackets'], df['apduLen']], axis=1)    # classic feature vector
    #dft = pd.concat([df['averageTimeDelta'], df['numOfPackets'], df['percentS'], df['percentI'], df['percentU'], df['Interro'],	df['Norm'],	df['SP']], axis=1) 
    #dft = pd.concat([df['averageTimeDelta'], df['apduLenRate']], axis=1)
    #dft = pd.concat([df['percentI'], df['percentU'], df['percentS']], axis=1)
    dft = pd.concat([df['numOfPackets'], df['apduLen']], axis=1)
    #dft = pd.concat([df.iloc[:, 7:12], df.iloc[:, 13:15]], axis=1)
    #dft = df.iloc[:, 9:12]
    #dft = pd.DataFrame(df.iloc[:, 14])
    #dft = pd.concat([df.iloc[:, 2], df.iloc[:, 5:8], df.iloc[:, 9:]], axis=1)  # select certain features
    #dft = pd.concat([df.iloc[:, 2], df.iloc[:, 6:]], axis=1)  # select certain features
    #dft = df.iloc[:,5:8]
    #dft = pd.concat([df.iloc[:, 2], df.iloc[:, 5], df.iloc[:, 7:9]], axis=1)
    #dft = pd.concat([df.iloc[:, 5], df.iloc[:, 7:9]], axis=1) # This is the classic feature vector used in previous submissions
    #dft = pd.concat([df.iloc[:, 2], df.iloc[:, 7:12]], axis=1)
    #dft = df.iloc[:, 7:9]
    print('*****************************\ncurrent feature vector is "%s" ' % list(dft.columns))
    print(dft.head())
    dft = dft.astype('float')
    #print(dft.dtype)
    dft_std = stats.zscore(dft)
    #print(np.isnan(dft_std))
    #print(np.isinf(dft_std))
    #print(np.isneginf(dft_std))
    print('***** standardize finished')
    #dft_std = dft_std.reshape(dft_std.size, 1)

    # create new folders to store output files with current time + parameters
    cur_time = str(time.strftime("%c", time.localtime()))
    out_subpath = cur_time + str(Ks) + str(infile)
    #print('new sub-folder=', out_subpath, 'will be created under output_path=', output_path)
    out_subpath = Path(output_path, out_subpath)
    print('new sub-folder for output files is created: ', out_subpath)
    out_subpath.mkdir(parents=True, exist_ok=True)
    #fn = "test.txt" 
    #filepath = p / fn
    #with filepath.open("w", encoding ="utf-8") as f:
    #    f.write(result)

    km = [KMeans(n_clusters=i) for i in Ks]
    isElbow = False
    distortions = []
    (distortions, isElbow) = elbow(km, dft_std)
    isInertia = False
    (inertias, isInertia) = inertia(km, dft_std)
    silhouette(Ks, km, dft_std, out_subpath)

    # plot elbow
    if isElbow == True:
        plt.plot(Ks, distortions)
        plt.xlabel('K = Number of Clusters')
        plt.ylabel('Average Euclidean Distance')
        plt.title('Elbow Method Resolving Optimal K')
        elbow_fig = Path(out_subpath, str(Ks) + str(infile) + 'Elbow_avg.eps')
        plt.savefig(elbow_fig)
        plt.show()


    # plot inertia
    if isInertia == True:
        plt.plot(Ks, inertias)
        plt.xlabel('K = Number of Clusters')
        plt.ylabel('Sum of Euclidean Distance')
        plt.title('Elbow Method Resolving Optimal K')
        inertia_fig = Path(out_subpath, str(Ks) + str(infile) + 'inertia_avg.eps')
        plt.savefig(inertia_fig)
        plt.show()

    sys.exit(0)

