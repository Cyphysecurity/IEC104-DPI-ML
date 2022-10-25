'''
Created by xxq160330 at 9/12/2018 8:53 PM
This is the top/main file for RTU focused analysis
Typical pipeline:
Before everything, for output measurement CSV directly from Java parser
    # merge files under the same src IP
    # e.g. 192.168.111.46;192.168.250.4.csv and 192.168.111.46;192.168.250.3.csv --> 192.168.111.46.csv
1. Function 3, create physical variables encoder from RTU list
2. Function 1, label data using encoder from step 1
3. other_stats.py -> cleanDF, needs to apply first to separate ASDU_Type-CauseTx
4. Rest steps in supervised learning pipeline
    4.1 Merge all labeled files into one big data file 'labeled_combined.csv' --> Function 4
    4.2 Calculate feature vector based on per RTU
        4.2.1 for each station, generate individual file --> Function 4
        4.2.2 for all stations, generate one file 'stats_combined.csv' --> Function 6
    4.3 Go through pipeline --> Function 5
'''
import pandas as pd
import numpy as np
from matplotlib.dates import epoch2num
import labeling as labeling
import utilities as utilities
import preprocessing as prep
import learning as learning
import plotting as plotting
import other_stats as ostats

import time
import sys
from os import listdir
from os import path

if __name__ == '__main__':
    # Before everything, for output measurement CSV directly from Java parser
    # merge files under the same src IP
    # e.g. 192.168.111.46;192.168.250.4.csv and 192.168.111.46;192.168.250.3.csv --> 192.168.111.46.csv
    # dPath = 'D:\PycharmProjects\XM\data1'
    # outPath = 'D:\PycharmProjects\XM\output1'
    # prep.mergeSameSrc(dPath, outPath)
    workPath = ""
    while True:
        print('*******************************\nChoose the function to process:\n')
        print('1. Label physical types\n'
              '2. Generate txt file for dictionary: codedict.txt '
              '(if dont need this text file or Python dictionary structure, this step doesnt need to be run\n'
              '3. Create encoder from RTU point list\n'
              '4. Calculate feature vector (Preprocess for machine learning or time modeling)\n'
              '5. Multi-class classification pipeline\n'
              '6. Calculate and collect statistics on labeled data\n'
              '# # 6.1. plots of time series and stats metrics per physical type;\n'
              '# # 6.2. stats_combined.csv     --statistical metrics;\n'
              '# # 6.3. stats for all physical types in a bar chart\n'
              '7. Each physical variable encoding in ASDU types\n'
              '8. Plot AGC\n'
              '0. Exit\n'
    
              '***********************************')
        choice = int(input('type the function number: '))
        start_time = time.time()
        # input: absolute path for input and output data
        # output: all labeled, xxx_labeled.csv, in "output" directory
        if choice == 1:
            dPath = 'D:\PycharmProjects\XM\data\AGC\sept13_2018_8pm'
            outPath = 'D:\PycharmProjects\XM\output\AGC\sept13_2018_8pm'
            files = [path.join(dPath, file) for file in listdir(dPath) if path.isfile(path.join(dPath, file))]
            for file in files:
                #infn = str(input('please input the measurement csv file for a specific substation. E.g. 192.168.111.1.csv'))
                infn = file
                indf = pd.read_csv(infn, delimiter=',')
                labeling.labeling(indf, infn, dPath, outPath)
            print('************* all labeling is finished!!! Cost %s seconds *****************' % (time.time() - start_time))
        elif choice == 2:
            #ref = pd.read_csv('physical_type_encoder.csv', delimiter=',')
            ref = pd.read_csv('encoder_draft.csv', delimiter=',')
            dictTxt = open('codedict.txt', 'w')
            labeling.createDict(ref, dictTxt)
        elif choice == 3:
            # input: RTUs_pointsvariables.xlsx
            # output: encoder_draft.csv
            rtuList = pd.read_excel('D:\PycharmProjects\XM\RTUs_PointsVariables.xlsx', sheet_name='Sheet1')
            rtuList.columns.str.strip()     # clean messy whitespaces in many elements
            df = pd.concat([rtuList.iloc[:, :2], rtuList.iloc[:, 3]], axis=1)
            df = df.rename(columns={'Type Variable': 'Type'}).to_csv('encoder_draft.csv')
            print('************* encoder_draft.csv has been generated!!! *******************')

        elif choice == 4:
            # input: labeled data
            # output: a total feature vector (unscaled and scaled) of all stations
            dPath = 'D:\PycharmProjects\XM\prepped\combined'
            outPath = 'D:\PycharmProjects\XM\prepped\stats'

            # step 1.0: take opposite value for negative measurement
            dPath1 = 'D:\PycharmProjects\XM\prepped\isNotNegative_combined'
            files = [path.join(dPath, file) for file in listdir(dPath) if path.isfile(path.join(dPath, file))]
            for file in files:
                df = pd.read_csv(file, delimiter=',')
                prep.takeOpposite(df)
                df.to_csv(path.join(dPath1, path.relpath(file, dPath)).replace('.csv', '_opposite.csv'))
                print('********** %s has been taken opposite values***************' % file)

            # step 1: merge individual labeled data into one csv
            # generate labeled_combined.csv
            prep.mergeLabeled(dPath1, outPath)

            # step 2: calculate fv for individual station CSV
            files = [path.join(dPath1, file) for file in listdir(dPath1) if path.isfile(path.join(dPath1, file))]
            dfl = []
            for file in files:
                #df = pd.read_csv('192.168.111.24_labeled.csv', delimiter=',')
                df = pd.read_csv(file, delimiter=',')
                #c = int(input('Please choose stats metrics computation per signature or per physical type:\n1 = Signature; 2 = Physical type'))
                fv = prep.calculate(df, 1)
                fv.to_csv(path.join(outPath, path.relpath(file, dPath)).replace('.csv', '_fv.csv'))
                print('********** %s has been scaled ***************' % file)
                dfl.append(fv)

            # step 3: combine fv from all individual IPs; generate a total fv and scale it
            fv_combined = pd.concat(dfl)
            fv_scaled = prep.scaling(fv_combined)
            #combined_fn = path.join(outPath, 'combined.csv')
            fv_combined.to_csv(path.join(outPath, 'combined.csv'))
            np.savetxt(path.join(outPath, 'scaled.csv'), fv_scaled, delimiter=',')

        elif choice == 5:
            #import src.rtu.multiclass_classification as pipeline
            #pipeline


            # INPUT file to learn, generated from choice 4
            df = pd.read_csv('D:\PycharmProjects\XM\prepped\stats\stats_combined.csv', delimiter=',')
            dft = prep.filterLabel(df)
            print('Physical labels are: ', dft.Physical_Type.unique())
            print('Src ip is: ', dft.srcIP.unique())
            dft = prep.mergeCurPow(dft)
            print('Physical labels are: ', dft.Physical_Type.unique())
            print('Verify columns in X: \n', dft.iloc[:2, 2:8])
            print('Verify column in y: \n', dft.iloc[:2, 8])

            # Generate X, y
            # scaling or not, run each learning algorithm
            x = dft.iloc[:, 2:8].astype(float).values
            x_scaled = prep.scaling(x)
            y = dft.iloc[:, 8].values
            
            x_train, x_test, y_train, y_test = learning.simpleSplit(x, y, 0.3, 5)
            accuracy, y_pred = learning.dtree(x_train, y_train, x_test, y_test)
            learning.confusionM(y_test, y_pred)
            learning.clfreport(y_test, y_pred)


            '''
            # Benchmark parameters, split percent and seed
            percents = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
            seeds = [1, 5, 10, 15, 20, 25, 30, 50]
            for p in percents:
                for s in seeds:
                    print('p = %f, s = %f' % (p, s))
                    x_train, x_test, y_train, y_test = learning.simpleSplit(x, y, p, s)
                    learning.dtree(x_train, y_train, x_test, y_test)
                    learning.gnb(x_train, y_train, x_test, y_test)
                    learning.knn(x_train, y_train, x_test, y_test)
            '''


            '''
            # kFolds, print out partitions
            cnt = 0
            for trainindex, testindex in learning.kfolds(x):
                print('TEST index:', testindex)
                x_train = [x[i] for i in trainindex]
                x_test = [x[i] for i in testindex]
                y_train = [y[i] for i in trainindex]
                y_test = [y[i] for i in testindex]
                addone = learning.dtree(x_train, y_train, x_test, y_test)
                cnt = cnt + addone
            print(cnt)
            '''

        elif choice == 6:
            # input: labeled data from choice 1
            # output:
            # # 1. plots of time series and stats metrics per physical type;
            # # 2. stats_combined.csv     --statistical metrics
            # # 3. stats for all physical types in a bar chart
            dPath = 'D:\PycharmProjects\XM\prepped\combined'
            outPath = 'D:\PycharmProjects\XM\prepped\stats'

            #c = int(input('Please choose stats metrics computation per signature or per physical type:\n'
                          #'1 = Signature; 2 = Physical type'))
            # get statistics for combined data frames
            #fv = prep.calculate(pd.read_csv(path.join(outPath, 'labeled_combined.csv'), delimiter=','), 2)
            fv = prep.calculate(pd.read_csv('D:\PycharmProjects\XM\prepped\stats\labeled_combined.csv', delimiter=','), 1)
            # separate 'srcIP', 'IOA', 'Physical_Type' from signature into new columns
            fv[['srcIP', 'IOA', 'Physical_Type']] = fv['Signature'].str.split(';', expand=True)
            fv['RTU'] = fv['srcIP'] + ';' + fv['IOA']

            # output 2
            fv.to_csv(path.join(outPath, 'stats_combined.csv'))

            '''
            # output 1
            df = pd.read_csv('D:\PycharmProjects\XM\prepped\\10_8\labeled_combined.csv', delimiter=',')
            grouped = df.groupby('Physical_Type')
            # plot time series for each phy type (key)
            for key in grouped.groups.keys():
                phy = grouped.get_group(key)
                plotting.ts_plot(phy['Time'], phy['Measurement'], key)
            #plotting.ts_plot(grouped.get_group('Frequ')['Time'], grouped.get_group('Frequ')['Measurement'], 'Frequ')

            # output 3
            labels = fv.index.tolist()      # labels in the following bar chart
            rows = np.empty([10, 6])        # a matrix, row = physical type, column = stats
            for i in np.arange(10):
                rows[i] = fv.loc[fv.index == labels[i]].values[0]
            plotting.statsBar1(labels, rows)
            '''
        elif choice == 7:
            # list of ASDU Types for this physical variable
            #infn = 'D:\PycharmProjects\XM\output\combined\\192.168.111.1_labeled.csv'
            dPath = 'D:\PycharmProjects\XM\output\AGC\sept13_2018_8pm'
            outPath = 'D:\PycharmProjects\XM\prepped\AGC\sept13_2018_8pm'
            files = [path.join(dPath, file) for file in listdir(dPath) if path.isfile(path.join(dPath, file))]
            outfiles = [file.replace(dPath, outPath) for file in files]
            for file in files:
                df = pd.read_csv(file, delimiter=',')
                if df.size == 0:
                    continue
                print('**************** working on file: ', file)
                dft = ostats.cleanDf(df)
                dft.to_csv(file.replace(dPath, outPath).replace('.csv', '_cleaned.csv'))
                # print out RTU count per physical type per station
                #dict1 = ostats.phyInASDUType(dft)
                #dict2 = ostats.phyCountPerIP(dft)


        elif choice == 8:
            # AGC plot, in 2018 capture
            #dpath = 'D:\PycharmProjects\XM\prepped\AGC\sept13_2018_8pm'
            dpath = 'D:\PycharmProjects\XM\prepped\AGC\sept10_2018_8am'
            files = [path.join(dpath, file) for file in listdir(dpath) if path.isfile(path.join(dpath, file))]
            dflist = []
            # AGC plot, in 2018 Sep.13, 8pm
            for f in files:
                print('File: %s' % f)
                df = pd.read_csv(f, delimiter=',')
                print('This File starts from %s, ends on %s' % (
                utilities.epochConverter(np.min(df.Time)), utilities.epochConverter(np.max(df.Time))))
                print('This file has the following physical types:', df.Physical_Type.unique())
                plist = ['P', 'not_exist', '-']
                # plist = ['P']
                ppower = df[df.Physical_Type.isin(plist)]
                if ppower.srcIP.unique()[0] == '192.168.111.96':
                    ioalist = [1037, 1070, 1072]
                    ppower = ppower[ppower.IOA.isin(ioalist)]
                print('plist filtered file has the following IOAs:', ppower.IOA.unique())
                dflist.append(ppower)
            plotting.plotagc(dflist)
            '''
            # AGC plot, in 2018 Sep.10
            for f in files:
                print('File: %s' % f)
                df = pd.read_csv(f, delimiter=',')
                print('This File starts from %s, ends on %s' % (utilities.epochConverter(np.min(df.Time)), utilities.epochConverter(np.max(df.Time))))
                print('This file has the following physical types:', df.Physical_Type.unique())
                plist = ['P', 'not_exist', '-']
                #plist = ['P']
                ppower = df[df.Physical_Type.isin(plist)]
                print('plist filtered file has the following IOAs:', ppower.IOA.unique())
                dflist.append(ppower)
            plotting.plotagc(dflist)
            '''
            '''
            # AGC plot, 111.96 in 2017 capture
            files = [path.join(dpath, file) for file in listdir(dpath) if path.isfile(path.join(dpath, file))]
            dflist = []
            for f in files:
                print('File: %s' % f)
                df = pd.read_csv(f, delimiter=',')
                print('This File starts from %s, ends on %s' % (
                utilities.epochConverter(np.min(df.Time)), utilities.epochConverter(np.max(df.Time))))
                print('This file has the following physical types:', df.Physical_Type.unique())
                if df.srcIP.unique()[0] == '192.168.111.96':
                    ppower = df[df.Physical_Type == 'P']
                    day4p = ppower[ppower.Time < 1501909200]  # '2017-08-05 00:00:00'
                    day8p = ppower[ppower.Time.between(1502168400, 1502254800)]
                    day9p = ppower[ppower.Time.between(1502254800, 1502341200)]
                    day10p = ppower[ppower.Time.between(1502341200, 1502427600)]
                    day11p = ppower[ppower.Time.between(1502427600, 1502514000)]
                    dfList = [day4p, day8p, day9p, day10p, day11p]
                else:
                    dfList.append(df)
            plotting.plotagc(dfList)
            '''
            '''
            # AGC plot, 111.24 in 2017 capture
            f = path.join(dpath, '192.168.111.24_labeled_cleaned.csv')
            print('File under plotting now is %s' % f)
            df = pd.read_csv(f, delimiter=',')
            print('This File starts from %s, ends on %s' % (utilities.epochConverter(np.min(df.Time)), utilities.epochConverter(np.max(df.Time))))
            #df['Time'] = pd.to_datetime(df['Time'], unit='s')   # convert epoch to readable time
            ppower = df[df.Physical_Type == 'P']
            day4p = ppower[ppower.Time < 1501909200]    # '2017-08-05 00:00:00'
            #day8p = ppower[ppower.Time < 1502254800]    # '2017-08-09 00:00:00'
            #day8p = day8p[day8p.Time > 1502168400]      # '2017-08-08 00:00:00'
            day8p = ppower[ppower.Time.between(1502168400, 1502254800)]
            day9p = ppower[ppower.Time.between(1502254800, 1502341200)]
            day10p = ppower[ppower.Time.between(1502341200, 1502427600)]
            day11p = ppower[ppower.Time.between(1502427600, 1502514000)]
            dfList = [day4p, day8p, day9p, day10p, day11p]

            print('Power measurements start from %s, ends on %s' % (np.min(ppower.Time), np.max(ppower.Time)))
            plotting.plotagc(dfList)
            '''
        elif choice == 0:
            print('exiting now...')
            sys.exit(0)

        else:
            print('Unknown choice!!!')
