'''
Created by xxq160330 at 10/1/2018 8:44 PM
This script calculates various types of statistics from data
which may be trivial to include in other scripts
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# input: cleaned dataframe of labeled data
# output: a dictionary (will print to screen as well)
# # key = physical var, val = list of ASDU Types for this physical variable
def phyInASDUType(df):
    grouped = df.groupby(by='Physical_Type')['ASDU_Type']
    groupNum = grouped.ngroups
    asduTypePerPhy = {}
    print('For each physical variable, it is encoded in the following ASDU types:')
    np.set_printoptions(suppress=True)      # suppress scientific notation
    for phy in grouped.groups.keys():
        x = grouped.get_group(phy)
        asduTs = x.unique()     # contains all the unique ASDU types for this physical variable
        print(phy, ': ', asduTs)
        #print(asduTs)
        asduTypePerPhy[phy] = asduTs

    return asduTypePerPhy

# input: cleaned dataframe of labeled data
# output: for each IP, list and count of all IOAs per physical variable
def phyCountPerIP(df):
    groupedIOA = df.groupby('Physical_Type')['IOA']
    ioaListPerPhy = {}
    print('For each physical variable, it has been reported through the following IOAs:')
    for phy in groupedIOA.groups.keys():
        ioaList = groupedIOA.get_group(phy).unique()
        ioaCnt = groupedIOA.get_group(phy).unique().size
        print(phy, ': ', ioaList, ', cnt = ', ioaCnt)
        ioaListPerPhy[phy] = ioaList
    return ioaListPerPhy

# input: original dataframe of labeled data, with column 'ASDU_Type-CauseTx'
# output: cleaned labeled data, 'ASDU_Type-CauseTx' separated as two individual columns
def cleanDf(df):
    tct = df['ASDU_Type-CauseTx']
    (asduType, causetx) = (tct.str.split(pat='-', expand=True).loc[:, 0], tct.str.split(pat='-', expand=True).loc[:, 1])
    df_separated = pd.concat([df, asduType, causetx], axis=1).rename(columns={0: 'ASDU_Type', 1: 'CauseTx'})
    dff = pd.concat([df_separated['srcIP'], df_separated['dstIP'], df_separated['ASDU_Type'], df_separated['IOA'],
                     df_separated['CauseTx'], df_separated['Time'], df_separated['Measurement'],
                     df_separated['Physical_Type'], df_separated['Signature']], axis=1)
    print('Cleaning ASDU_Type-CauseTx finished...')
    return dff

