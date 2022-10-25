'''
Created by grace at 5/29/18 9:16 PM
This script will calculate and plot analysis of malformed packets
''' 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import time
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy import stats


def convertEpoch(epoch):
    #return time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime(epoch))
    return time.strftime("%d %b %Y ", time.localtime(epoch))

# plot avg pkt capture rate in five datasets
def pktRateInTime(df):
    dft = pd.concat([df.iloc[:, 0:2], df.iloc[:, 3], df.iloc[:, 6]], axis=1)
    dft['firstTimeSeen'] = df['firstTimeSeen'].apply(
        lambda x: convertEpoch(x))  # convert epoch time to readable date&time
    # rint(dft)
    dft['flow'] = dft['srcIP'] + '->' + dft['dstIP']  # create flow label
    dft = dft.iloc[:, 2:]
    print(dft)
    gb = dft.groupby('flow')
    flowdict = gb.groups
    groups = [gb.get_group(x) for x in flowdict]  # split the original dataframe into one dataframe per flow
    for i in range(0, len(groups)):
        curFlow = groups[i]
        legend = curFlow['flow'].iloc[0]  # use flow name as plot legend
        if curFlow.shape[0] == 1:
            print(legend + ': only has one sample')
        elif curFlow.shape[0] > 1:
            error = np.std(curFlow['packetCaptureRate'])
            mean = np.mean(curFlow['packetCaptureRate'])
            plt.figure(figsize=(7, 5))
            plt.bar(curFlow['firstTimeSeen'], curFlow['packetCaptureRate'], yerr=error, align='center', width=0.5)
            plt.axhline(y=mean, color='red', linestyle='--')  # add horizontal line to show average packet capture rate
            plt.xlabel('Date and time')
            plt.ylabel('Number of packets per second')
            plt.xticks(curFlow['firstTimeSeen'], rotation=45)
            plt.tight_layout()
            plt.savefig('malformed_flow_timeseries.svg', bbox_inches='tight')
            plt.grid()
            plt.show()

# plot distribution of IOA in malformed pkts vs good pkts
def distributionIOA():
    dfm = pd.read_csv('malformedIOA.csv', delimiter=',')
    dfg = pd.read_csv('distinctIOA_total.csv', delimiter=',')
    df_new = dfm.append(dfg)
    df_new = pd.to_numeric(df_new['ioa'])
    print(df_new.dtypes)

    colors = ['navy', 'tomato']

    #cnt = 0
    #for i in range(0, df_new.shape[0]):
    #    plt.scatter(df_new.iloc[i,0], df_new.iloc[i,1], color=colors[int(df_new.iloc[i,1])])
    #    print(cnt)
    #    cnt += 1
    x = df_new.iloc[:,0]
    y = df_new.iloc[:,1]
    col = np.where(y == 0, 'b', 'r')
    plt.scatter(x, y, c=col)
    plt.xlabel('IOA')
    plt.ylabel('1 = malformed, 0 = good')
    plt.grid(True)
    plt.savefig('malformedIOA.png')
    plt.show()
    print('finished!!!')

if __name__ == '__main__':
    df = pd.read_csv('totalMalformProcessed.csv', delimiter=',')

    #pktRateInTime(df)
    distributionIOA()
    epoch = 1502291879.77114
    print(convertEpoch(epoch))


