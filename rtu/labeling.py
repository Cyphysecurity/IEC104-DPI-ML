'''
Created by xxq160330 at 9/6/2018 4:47 PM
This script labels the measurement CSV files with physical variable types, such as voltages, power, frequency, etc.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os import path

import re
import sys

# label each data point in measurement time series with physical types
# add a new column: physical_type
# output: xxx.xxx.xxx.xxx_labeled.csv
def labeling(indf, infn, dPath, outPath):
    ######### generate dictionary structure as encoder for signature check
    ref = pd.read_csv('encoder_draft.csv', delimiter=',')
    keys = ref.iloc[:, 1].astype(str) + ';' + ref.iloc[:, 2].astype(str)
    vals = ref.iloc[:, 3].astype(str).apply(str.strip)
    codedict = {}
    for k, v in zip(keys, vals):
        codedict[k] = v
    ###############
    outfn = path.join(outPath, path.relpath(infn, dPath).replace('.csv', '_labeled.csv'))
    #outfn = infn.replace('.csv', '_labeled.csv')
    print('***************** %s is under labeling... *********************' % infn)
    outfile = open(outfn, 'w', newline='')
    outdf = indf.apply(signatureCheck, args=(codedict,), axis=1)       # perform signature test on each row
    outdf.to_csv(outfile)
    print('***************** Labeled file %s generated!!! *********************' % outfn)

# read in dict TXT, match physical types, add as a new column
def signatureCheck(df, codedict):
    #dictf = open('codedict.txt', 'r')

    #sig = df.loc['srcIP'] + '-' + df.loc['dstIP'] + '-' + str(df.loc['ASDU_Type']) + '-' + str(df.loc['IOA'])
    sig = df.loc['srcIP'] + ';' + str(df.loc['IOA'])
    df.loc['Signature'] = sig
    if sig in codedict.keys():
        df.loc['Physical_Type'] = codedict[sig]
        df.loc['Signature'] = sig + ';' + codedict[sig]
        #print('sig is: ', sig, 'Physical type = ', df.loc['Physical_Type'])
    else:
        df.loc['Physical_Type'] = 'not_exist'
    return df

# input: encoder CSV
# output: dictionary TXT, line = key:val
def createDict(ref, dictTxt):
    #keys = ref.iloc[:, 0] + '-' + ref.iloc[:, 1] + '-' + ref.iloc[:, 2].astype(str) + '-' + ref.iloc[:, 4].astype(str)
    #vals = ref.iloc[:, 5]
    keys = ref.iloc[:, 1].astype(str) + '-' + ref.iloc[:, 2].astype(str)
    vals = ref.iloc[:, 3].astype(str).apply(str.strip)
    codedict = {}
    for k, v in zip(keys, vals):
        codedict[k] = v
        dictTxt.write(k + ':' + v + '\n')
    dictTxt.close()
    print('************* codedict.txt has been generated!!! *******************')
    return codedict

if __name__ == '__main__':
    print('*******************************\nChoose the function to process:\n')
    print('1. label physical types\n2. generate txt file for dictionary: codedict.txt (if dont need this text file '
          'or Python dictionary structure, this step doesnt need to be run\n'
          '3. create encoder from RTU point list\n'
          '***********************************')
    choice = int(input('type the function number: '))
    if choice == 1:
        infn = str(input('please input the measurement csv file for a specific substation. E.g. 192.168.111.1.csv'))
        indf = pd.read_csv(infn, delimiter=',')
        labeling(indf, infn)
    elif choice == 2:
        #ref = pd.read_csv('physical_type_encoder.csv', delimiter=',')
        ref = pd.read_csv('encoder_draft.csv', delimiter=',')
        dictTxt = open('codedict.txt', 'w')
        createDict(ref, dictTxt)
    elif choice == 3:
        # input: RTUs_pointsvariables.xlsx
        # output: encoder_draft.csv
        rtuList = pd.read_excel('RTUs_pointsvariables.xlsx', sheet_name='Sheet1')
        rtuList.columns.str.strip()     # clean messy whitespaces in many elements
        df = pd.concat([rtuList.iloc[:, :2], rtuList.iloc[:, 3]], axis=1)
        df = df.rename(columns={'Type Variable': 'Type'}).to_csv('encoder_draft.csv')
        print('************* encoder_draft.csv has been generated!!! *******************')
