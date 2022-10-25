'''
Created by xxq160330 at 5/15/2018 3:12 PM
This script takes in extracted data from Java code
and prepossesses for further steps
input: txt file, "," delimited
output: csv file
'''

import csv
import pandas as pd
import numpy as np
from os.path import isfile
import pathlib
from pathlib import Path, WindowsPath

cur_p = Path()
in_p = Path(cur_p.absolute(), 'input')
out_p = Path(cur_p.absolute(), 'output')

# tuple = <srcIP, average delta time, direction, # times reported>
def preprocess1(csvwriter, lines):
    #csvwriter.writerow(['srcIP', 'averageTimeDelta', 'direction','numTimesReported'])
    #csvwriter.writerow(['averageTimeDelta', 'direction','numTimesReported'])
    #csvwriter.writerow(['direction','numTimesReported'])
    csvwriter.writerow(['averageTimeDelta', 'numTimesReported'])
    print('header written')
    for line in lines:
        fields = line.split(',')
        (ip, t, direction, occurrence) = (fields[0].split(':')[1], float(fields[1].split(':')[1]), fields[2].split(':')[1], float(fields[3].split(':')[1]))
        if (direction == 'S->C'):
            direction = 1.0
        else:
            direction = 0.0

        csvwriter.writerow(list((direction, occurrence)))

# write header and data
def preprocess2(csvwriter, lines):
    header = [item.split(':')[0] for item in lines[0].split(',') if item]
    #csvwriter.writerow(header[:-2])
    csvwriter.writerow(header)
    #print(header)
    print('header written...')
    print(lines)
    for line in lines:
        #print(line)
        row = [item.split(':')[1] for item in line.split(',') if item]
        #encoded_row = encodeIP(row)
        #csvwriter.writerow(row[:-2])
        csvwriter.writerow(row)
        #csvwriter.writerow(encoded_row)

# write data only, suppose file of the same header already exists
def preprocess3(csvwriter, lines):
    for line in lines:
        print(line)
        row = [item.split(':')[1] for item in line.split(',') if item]
        #csvwriter.writerow(row[:-2])
        csvwriter.writerow(row)

# extract malformed IOA and write out to a csv
def writeMalformedIOA(csvwriter, lines):
    ioaSet = set()
    for line in lines:
        print(line)
        ioaList = line.split(',')[-2].split(':')[1].split(' ')
        for ioa in ioaList:
            if ioa not in ioaSet:
                ioaSet.add(ioa)
    for ioa in ioaSet:
        csvwriter.writerow([ioa])

# encode IPs
# power grid example:
# 192.168.250.x -> C1, C2, C3, C4
# 192.168.111.y -> according to order by y, assign O1 to O38
# gas plant example:
# 171.31.3.100 -> C1
# 171.31.3.96 -> S96
def encodeIP(dict_f, line, dataset):
    # generate a dictionary of IPs and their corresponding codes
    df = pd.read_csv(dict_f, sep=',', encoding='utf_8')
    keys = df.iloc[:, 0]
    codedict = {}  # dict, key = ip, val = code
    if dataset == 'bulk':
        keys = str(keys)

    #elif dataset == 'gas':
    #    vals = df.iloc[:, 1].astype(str)
    vals = df.iloc[:, 1].astype(str)
    for k, v in zip(keys, vals):
        codedict[k] = v
    print(codedict)

    # encode IP in this line
    flds = line.split(',')  # list of all fields in one sample
    print('original fields:\n', flds)
    for i in range(0, len(flds)):
        if flds[i].startswith('srcIP') | flds[i].startswith('dstIP') | flds[i].startswith('rtu') | flds[i].startswith('server'):
            field = flds[i].split(':')
            (feature, val) = (field[0], field[1])
            code = codedict[val]        # find code through ip in the dictionary
            print('new coded ip = ', code)
            if feature == 'rtu':
                code = 'S' + code
            flds[i] = feature + ':' + code
    print('new fields:\n', flds)
    line = ','.join(flds)
    print('new line:\n', line)
    return line


if __name__ == '__main__':
    print('*******************************\nChoose the function to process:\n')
    print('1. malformed packets\n2. prepare clustering data\n3. test encodeIP\n \
        4. generate all outstation IPs\n5. encode IPs\n \
            ***********************************')
    choice = int(input('type the function number: '))
    if choice == 1:
        print('Preprocessing malformed packets......\n')
        with open('4_8malformed.txt', 'r') as txtfile:
            lines = list(line for line in (l.rstrip() for l in txtfile.readlines()[1:]) if line)  # ignore blank lines
        #print(lines)
        for i in range(0, len(lines)):
            newline = encodeIP(lines[i])
            print(newline)
            lines[i] = newline

        if isfile('malformProcessed.csv'):
            print('malformProcessed.csv already exists!!! Will append data to this file\n')
            csvfile = open('malformProcessed.csv', 'a', newline='')     # if this output file already exists, simple add new rows
            csvwriter = csv.writer(csvfile)
            preprocess3(csvwriter, lines)
            csvfile.close()
        else:
            print('malformProcessed.csv not exists, creating new file......\n')
            csvfile = open('malformProcessed.csv', 'w', newline='')     # if not exist, create this output file
            csvwriter = csv.writer(csvfile)
            preprocess2(csvwriter, lines)
            csvfile.close()

        choice2 = input('Do you want to generate malformed IOA list now? yes or no\n')
        if choice2 == 'no':
            print('exit this function now...')
        elif choice2 == 'yes':
            print('generating malformed IOA list......')
            csvfile = open('malformedIOA.csv', 'a', newline='')
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['malformedIOA'])
            writeMalformedIOA(csvwriter, lines)

            csvfile.close()

    elif choice == 2:
        print('Preprocessing clustering data / or any .txt data......')
        #txtfn = '../input/part1-clustering-feature.txt'
        #txtfn = '../input/IEC104-Analysis_104only_2018.json2020-04-18_00_46_36.txt'
        #txtfn = '../input/clustering-XM/IEC104-Analysis_8pm-sep13.txt'
        #txtfn = '../input/clustering-gas/104JavaParser_2020-10-27_23_42_18_104only-part31.txt'
        #txtfn = 'encoded_104JavaParser_2021-03-17_01_45_10_104only-0706-to-0710.txt'
        #txtfn = '0706-to-0710-ASDUType-IOA-time-series.txt'
        txtfn = '104JavaParser_2021-03-17_01_45_10_104only-0706-to-0710.txt'
        #txtfn = '104JavaParser_2020-12-07_17_29_29_power_dist.txt'
        in_file = Path(in_p, 'clustering-gas', txtfn)
        #in_file = Path(in_p, txtfn)
        #in_file = Path(input_path, 'power-dist', txtfn)
        print('file: ', txtfn)
        with open(in_file, 'r') as txtfile:
        # with open('/home/grace/PycharmProjects/iccp/combinedClustering.txt', 'r') as txtfile:
            lines = list(line for line in (l.rstrip() for l in txtfile.readlines()[1:]) if line)  # ignore blank lines
            # The line below used to be a little workaround for sysList comma problem
            # lines = list(line[:line.find('sysList')] for line in (l.rstrip() for l in txtfile.readlines()[1:]) if line)  # ignore blank lines
        
        csvfile = open(str(in_file).replace('.txt', '.csv'), 'w', newline='')                # clustering data needs to be written separately for each dataset
        csvwriter = csv.writer(csvfile)

        preprocess2(csvwriter, lines)
        csvfile.close()
    elif choice == 3:
        print('\n\n testing encoding IP......\n')
        with open('104JavaParser_2020-12-15_22_47_42_104only-07_20.txt', 'r') as txtfile:
            lines = list(line for line in (l.rstrip() for l in txtfile.readlines()[1:]) if line)
        print(lines[0])
        f = open('ip_station_encoder.csv', 'r')
        encodeIP(f, lines[0], 'gas')
    elif choice == 4:
        print('start creating list of distinct outstation IPs......')
        outIP = set()      # store all distinct outstation IPs
        with open('combinedProcessed.csv', 'r') as txtfile:
            lines = list(line for line in (l.rstrip() for l in txtfile.readlines()[1:]) if line)
        for line in lines:
            flds = line.split(',')
            srcIP = flds[0]
            dstIP = flds[1]
            #print('src = %s, dst = %s' % (srcIP, dstIP))
            if srcIP not in outIP:
                outIP.add(srcIP)
            if dstIP not in outIP:
                outIP.add(dstIP)
        print('the set of IPs:\n', outIP)
        print('there are %d IPs' % len(outIP))
        l = list(outIP).sort()
        print(l)
        #csvfile = open('ip.csv', 'w', newline='')
        #csvwriter = csv.writer(csvfile, delimiter=',')
        #csvwriter.writerow(['IP'])
        i = 0
        #for ip in outIP:
        #    csvwriter.writerow([ip])
            #print('%d. ip = %s, written...' % (i, ip))
            #i += 1
        #csvfile.close()
    elif choice == 5:
        print('Encoding all IPs into Cx, Oy......\n')
        txtfn = '104JavaParser_2021-03-17_01_45_10_104only-0706-to-0710.txt'
        in_file = Path(in_p, txtfn)
        with open(in_file, 'r') as txtfile:
            lines = list(line for line in (l.rstrip() for l in txtfile.readlines()[1:]) if line)  # ignore blank/header lines
        # print(lines)
        outfile = open(Path(in_p, 'encoded_' + txtfn), 'w', newline='')
        f = Path(cur_p.absolute(), 'ip_station_encoder.csv')
        newlines = []
        for i in range(0, len(lines)):
            print('line curren: \n', lines[i])
            newline = encodeIP(f, lines[i], 'gas')
            newline += '\n'
            print(newline)
            newlines.append(newline)

        outfile.writelines(newlines)
        outfile.close()

    print('finished!')

    
        