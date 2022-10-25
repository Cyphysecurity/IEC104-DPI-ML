'''
Created by xxq160330 at 9/12/2018 9:09 PM
This script preprocesses data for machine learning or time series modeling
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skpreprocessing
from os import listdir
from os import path

# input: labeled data, e.g. 111.x_labeled.csv
# output: dataframe, features per signature:
# numMeasure, mean, min, max, var, ACF
def calculate(df, c):
    fv = pd.DataFrame(columns=['mean', 'min', 'max', 'var', 'count', 'autocorr'])
    group_base = ''
    if c == 1:
        group_base = group_base.join('Signature')
    elif c == 2:
        group_base = group_base.join('Physical_Type')

    if group_base in df.columns:
        grouped = df.groupby(group_base)
        measure = grouped['Measurement']
        # feature vector: fv
        # index = signature, [mean, min, max, std, count, acf]
        fv = (measure.agg([np.mean, np.min, np.max, np.var])).rename(columns={'amin': 'min', 'amax': 'max'})
        fv['count'] = measure.count()
        fv['autocorr'] = measure.apply(pd.Series.autocorr)

        fv = fv.fillna(1)   # replace NaN with 1, unvarying data generates acf value of NaN
    print('************* stat metrics calculation based on ', group_base, 'already finished!!! *************')
    return fv


# input: data frames of feature vector
# output: scaled fv data frames
def scaling(df):
    return skpreprocessing.MinMaxScaler().fit_transform(df)
    #return preprocessing.RobustScaler().fit_transform(df)

# INPUT: absolute data path of all labeled data targeted to merge
# OUTPUT: one big file of all merged labeled data
def mergeLabeled(dPath, outPath):
    files = [path.join(dPath, file) for file in listdir(dPath) if path.isfile(path.join(dPath, file))]
    outf = open(path.join(outPath, 'labeled_combined.csv'), 'w')
    outdf = []
    for file in files:
        df = pd.read_csv(file, delimiter=',')
        outdf.append(df)
    pd.concat(outdf).to_csv(outf)
    print('********** merge of all labeled data has been finished! ***********')
    #return pd.concat(outdf)

# Filter '-' and 'nan' labels; remember excluding malformed station 111.97
# INPUT: data frames with calculated stats features
def filterLabel(df):
    df = df[df['srcIP'] != '192.168.111.97']
    phyList = ['P', 'Q', 'U', 'I', 'Status', 'Frequ', 'AGC-SP (Set-Point)', 'I-A', 'I-B', 'I-C', 'TapPosMv']
    #phyList = ['P', 'Q', 'U', 'I', 'I-A', 'I-B', 'I-C']
    dft = df[df['Physical_Type'].isin(phyList)]
    return dft

# Merge labels: 'I-A', 'I-B', 'I-C' -> 'I-three'; 'P', 'Q' -> 'Power'
def mergeCurPow(df):
    iList = ['I-A', 'I-B', 'I-C']
    pList = ['P', 'Q']
    #dft = df[df['Physical_Type'].isin(iList)]
    #print(dft['Physical_Type'].unique())
    df.Physical_Type[df.Physical_Type.isin(iList)] = 'I-three'
    print('After merging current labels, labels being classified are: ', df['Physical_Type'].unique())
    df.Physical_Type[df.Physical_Type.isin(pList)] = 'Power'
    print('After merging power labels, labels being classified are: ', df['Physical_Type'].unique())
    return df

# Make negative values in 'Measurement' positive values
# change in place
def takeOpposite(df):
    mask = (df['Measurement'] < 0)
    df_valid = df[mask]
    df.loc[mask, 'Measurement'] = np.negative(df_valid['Measurement'])

# input: original dataframe of labeled data, with column 'ASDU_Type-CauseTx'
# output: cleaned labeled data, 'ASDU_Type-CauseTx' separated as two individual columns
def cleanDf(df:pd.DataFrame):
    tct = df['ASDU_Type-CauseTx']
    (asduType, causetx) = (tct.str.split(pat='-', expand=True).loc[:, 0], tct.str.split(pat='-', expand=True).loc[:, 1])
    df_separated = pd.concat([df, asduType, causetx], axis=1).rename(columns={0: 'ASDU_Type', 1: 'CauseTx'})
    dff = pd.concat([df_separated['srcIP'], df_separated['dstIP'], df_separated['ASDU_Type'], df_separated['IOA'],
                     df_separated['CauseTx'], df_separated['Time'], df_separated['Measurement'],
                     df_separated['Physical_Type'], df_separated['Signature']], axis=1)
    print('Cleaning ASDU_Type-CauseTx finished...')
    return dff