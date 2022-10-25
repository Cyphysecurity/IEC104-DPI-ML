'''
This script is to try hierarchical clustering
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import stats
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as hiclustering
import sys, argparse, time, pathlib
from pathlib import Path
from argparse import ArgumentParser


def plot_dendrogram(df, output_path):
    print('Plotting dendrogram...')
    plt.figure(figsize=(12,8))
    plt.title('Initial Dendrogram')
    den = hiclustering.dendrogram(hiclustering.linkage(df, method='ward'))
    plt.tight_layout()
    fig_name = 'dendrogram.eps'
    plt.savefig(Path(output_path, fig_name))
    plt.show()

def agglo_clustering(df, output_path):
    agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    agg.fit_predict(df)
    plt.figure(figsize=(10, 8))
    plt.scatter(df[ : , 0], df[ : , 1], c=agg.labels_, cmap='rainbow')
    fig_name = 'agglo.eps'
    plt.savefig(Path(output_path, fig_name))
    plt.show()

if __name__ == '__main__':
    # show working directory path, input path, output path
    print('Please follow this format: python src/hierarchical.py @hierarchical-args.txt; modify the arguments in this txt file')
    print('Current path: ', Path().absolute())
    input_path = Path(Path().absolute(), 'input')
    output_path = Path(Path().absolute(), 'output')
    print('INPUT path: ', input_path)
    print('OUTPUT path: ', output_path)

    # parse command-line arguments
    cl_parser = ArgumentParser(fromfile_prefix_chars='@')
    cl_parser.add_argument('--infile')
    namespace = cl_parser.parse_args()
    print(namespace)
    infile = namespace.infile
    print('input data file: ', infile)

    # getting data
    #df = pd.read_csv('../output/IEC104-Analysis_part1_104only.json2020-04-09_19_35_12.csv', delimiter=',') # Netherland, tried feature vector: [9:12]
    #df = pd.read_csv('../input/IEC104-Analysis_104only_2018.json2020-04-18_00_46_36.csv', delimiter=',') # 2018 XM data
    #df = pd.read_csv(Path(input_path, 'combTypesProcessed.csv'), delimiter=',') # 2017 XM data
    #df = pd.read_csv(Path(input_path, 'clustering-XM', infile), delimiter=',') # 2017 XM data
    #df = pd.read_csv(Path(input_path, 'clustering-gas', infile), delimiter=',') # gas part 31 data
    df = pd.read_csv(Path(output_path, infile), delimiter=',')
    print(df.columns)
    #dft = pd.concat([df['averageTimeDelta'], df['numOfPackets'], df['apduLen']], axis=1)    # classic feature vector
    #dft = pd.concat([df['averageTimeDelta'], df['numOfPackets'], df['percentS'], df['percentI'], df['percentU'], df['Interro'],	df['Norm'],	df['SP']], axis=1) 
    #dft = pd.concat([df['averageTimeDelta'], df['apduLenRate']], axis=1)
    #dft = pd.concat([df['percentI'], df['percentU'], df['percentS']], axis=1)
    #dft = pd.concat([df['numOfPackets'], df['apduLen']], axis=1)
    dft = df
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
    cur_time = str(time.strftime("%b-%d-%Y-%H-%M-%S", time.localtime()))
    out_subpath = cur_time + str(infile)[:-5] + '_hierarchy'
    #print('new sub-folder=', out_subpath, 'will be created under output_path=', output_path)
    out_subpath = Path(output_path, out_subpath)
    print('new sub-folder for output files is created: ', out_subpath)
    out_subpath.mkdir(parents=True, exist_ok=True)
    #fn = "test.txt" 
    #filepath = p / fn
    #with filepath.open("w", encoding ="utf-8") as f:
    #    f.write(result)
    plot_dendrogram(dft_std, out_subpath)
    #print(dft_std)
    agglo_clustering(dft_std, out_subpath)
