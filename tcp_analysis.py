'''
This script analyze TCP flow (srcIP, dstIP, src port, dst port)
Use pkt2flow to extract TCP flows 
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, argparse, time, pathlib
from pathlib import Path
from argparse import ArgumentParser 

if __name__ == "__main__":
# show working directory path, input path, output path
    #print('Please follow this format: python findingK.py @findingK-args.txt; modify the arguments in this txt file')
    print('Current path: ', Path().absolute())
    subinput = 'merged_2017_tcp'
    input_path = Path(Path().parent.parent.absolute(), 'input', subinput)
    output_path = Path(Path().parent.parent.absolute(), 'output')
    print('INPUT path: ', input_path)
    print('OUTPUT path: ', output_path)

    totalDf = []
    # detect SYN-ACK, FIN-ACK/RST
    flagDict = {'0x00000010': 'ACK', '0x00000002': 'SYN', '0x00000012': 'SYN-ACK', \
        '0x00000001': 'FIN', '0x00000011': 'FIN-ACK', '0x00000004': 'RST', '0x00000014': 'RST-ACK', '0x00000008': 'PUSH', '0x00000018': 'PUSH-ACK'}
    for infile in input_path.rglob('*'):
       # infile = Path(input_path, 'aug11-104flows', 'tcp_syn', '192.168.250.2_40112_192.168.111.76_2404_1502464904.pcap.csv')
        print(infile)
        df = pd.read_csv(infile, delimiter=',')
        print('df dimension: ', df.shape)
        df = df.rename(columns={'ip.src': 'ip_src', 'ip.dst': 'ip_dst', 'tcp.srcport': 'tcp_srcport', 'tcp.dstport': 'tcp_dstport', 'tcp.flags': 'tcp_flags', 'frame.time_epoch': 'time_epoch'})
        print(df.head(5))
        df['flagName'] = df['tcp_flags'].map(flagDict)
        #print(df.head(9))

        #outdf = pd.DataFrame(columns=['ip_src', 'ip_dst', 'tcp_scrport', 'tcp_dstport', 'flagName', 'time_duration', 'startOrEnd'])
        outRows = []
        #outdf = pd.DataFrame(columns=df.columns)
        #print('outdf', outdf.columns)
        flagCnt = {}
        startT, endT = 0, 0
        for row in df.itertuples(index=True, name='Pandas'):
            #print(row)
            if row.flagName == 'SYN-ACK' and 'SYN-ACK' not in flagCnt:
                flagCnt['SYN-ACK'] = 1
                startT = row.time_epoch
                outRows.append(row)
            if row.flagName == 'SYN' and 'SYN' not in flagCnt:
                flagCnt['SYN'] = 1
                startT = row.time_epoch
                outRows.append(row)
            elif row.flagName == 'FIN-ACK' and 'FIN-ACK' not in flagCnt:
                flagCnt['FIN-ACK'] = 1
                endT = row.time_epoch
                outRows.append(row)
            elif row.flagName == 'RST' and 'RST' not in flagCnt: # only count RST when no FIN before it
                flagCnt['RST'] = 1
                if 'FIN-ACK' not in flagCnt:
                    endT = row.time_epoch
                    outRows.append(row)
            elif row.flagName == 'RST-ACK' and 'RST-ACK' not in flagCnt: # only count RST-ACK when no FIN before it
                flagCnt['RST'] = 1
                if 'FIN-ACK' not in flagCnt:
                    endT = row.time_epoch
                    outRows.append(row)
            elif row.flagName not in flagCnt:
                flagCnt[row.flagName] = 1
            else:
                flagCnt[row.flagName] += 1
        print(flagCnt)
        outdf = pd.DataFrame(outRows)
        if startT > 0 and endT > 0:
            outdf['duration'] = endT - startT   # redundant rows with the same duration, should only count one duration for repeated values
        print(outdf)
        totalDf.append(outdf)
    
    finalDf = pd.concat(totalDf)
    finalDf.to_csv(path_or_buf=Path(output_path, subinput + 'tcp_syn_short.csv'), sep=',')
        
    