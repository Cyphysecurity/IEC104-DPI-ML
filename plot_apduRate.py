'''
Created by xxq160330 at 6/19/2018 6:39 PM
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import sys

def plotting(df):
    print('************** start plotting **************\n')
    plt.figure(figsize=(10, 10))
    fig = plt.plot()
    colors = ['navy', 'tomato', 'turquoise', 'darkorange', 'red', 'orange', 'plum', 'grey', 'olive', 'brown']
    markers = ['^', 's', 'o', 'd', 'x', '1', '2', '3', '4', '*']
    flows = df.flow.unique()
    numLegend = len(flows)
    typeid = df.asdu_type.unique()[0]
    target_rows = df[df['flow'] == flows[1]].index.values
    for i in target_rows:
        print('time = %s, rate = %s' % (df.loc[i, 'time'], df.loc[i, 'apdu_rate']))
        plt.scatter(df.loc[i, 'time'], df.loc[i, 'apdu_rate'])
    '''
    for color, f, label in zip(colors[0:numLegend], range(0,numLegend), markers[0:numLegend]):
        target_rows = df[df['flow'] == flows[f]].index.values
        for i in target_rows:
            print('time = %s, rate = %s' % (df.loc[i,'time'], df.loc[i,'apdu_rate']))
            plt.scatter(df.loc[i,'time'], df.loc[i,'apdu_rate'], color=color, alpha=.6, label='Flow: %s' % flows[f], marker=label)
    print('*************** scatter plotting finished *****************\n')
    plt.title('%d Flows with Central Station Sending I type APDUs, ASDU TypeID = %s' % (numLegend, typeid))
    plt.xlabel('Epoch time')
    plt.ylabel('APDU capture rate (number/second)')
    print('*************** start showing *****************\n')
    #handles, ls = plt.gca().get_legend_handles_labels()
    #by_label = OrderedDict(zip(ls, handles))
    #plt.legend(by_label.values(), by_label.keys())
    '''
    print('*************** scatter plotting finished *****************\n')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    #plt.savefig('apduRate.svg')
    plt.show()

if __name__ == '__main__':
    infile = open('apduRate_per_asduType.csv', 'r')
    # src,dst,asdu_type,time,apdu#,apdu_rate
    #df = pd.read_csv(infile, dtype={'src':object, 'dst': object, 'asdu_type': int, 'time': float, 'apdu#':float, 'apdu_rate': float})
    df = pd.read_csv(infile, dtype=object)
    df['flow'] = df['src'] + ',' + df['dst']    # combine src,dst into flow
    df_flow = df.iloc[:, 2:]

    dfc = df[df.src.str.contains('192.168.250.\d')]             # filter out src = central stations
    dfc_grouped = dfc.groupby(by='asdu_type', sort=True).groups        # group by types, a dict
    type50 = dfc_grouped['50']
    type100 = dfc_grouped['100']
    df50 = pd.DataFrame(df_flow, index=type50)
    #df50.to_csv('type50_apdu_rate.csv', sep=',')
    df100 = pd.DataFrame(df_flow, index=type100)
    df100.to_csv('type100_apdu_rate.csv', sep=',')

    #plotting(df50)
    sys.exit(0)





