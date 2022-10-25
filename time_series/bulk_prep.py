'''
## This script:
1. get all input files parsed from 104 parser
2. TSFRESH feature engineering
'''

import tsfresh
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import numpy as np
import pathlib, time, sys, re, datetime
from pathlib import Path, WindowsPath
import matplotlib.pyplot as plt
import seaborn as sb
import sktime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters

# Utility functions
def my_extraction(df, params):
    print('*** Start tsfresh extraction ***')
    print('There are {} keys in params'.format(len(params.keys())))
    return extract_features(df, column_id='id', column_sort='Time', default_fc_parameters=params)


def print_info(df):
    print('In this input data: \n')
    print('df shape: ', df.shape)
    print('RTUs/SCADA server: {} \nASDU types: {} \nNumber of RTUs: {}'.format( \
        df.dstIP.unique(), list(df['ASDU_Type'].unique()), len(df['ASDU_addr'].unique())))

if __name__ == '__main__':

    # I/O
    cur_p = Path().absolute()
    # G:\My Drive\IEC104\XM-results\IOA_timeseries\bulk2017
    in_p = Path('G:\My Drive', 'IEC104', 'XM-results', 'IOA_timeseries', 'bulk2017')
    out_p = Path(Path().absolute().parents[1], 'output')
    print('Current path: {}\nInput path: {}\nOutput path: {}'.format(cur_p, in_p, out_p))
    #sys.exit()

    df = pd.DataFrame()
    cnt = 0
    if not in_p.exists():
        print('Google Drive may not be active!!!')
        sys.exit()

    for f in in_p.glob('*'):
        #print(f)
        cur_df = pd.read_csv(f)
        cnt += cur_df.shape[0]
        df = pd.concat([df, cur_df])
    print('After concatenation, data row # equal to summing up all the individual dataframes? ', cnt==df.shape[0], df.shape[0])

    # Simple prep
    df['id'] = df['srcIP'].astype(str) + '-' + df['IOA'].astype(str)
    print_info(df)

    # Step 1: divide data into analog/digital variables based on ASDU types
    # ASDU types: [13, 36, 3, 31, 9, 50, 30, 1, 5]
    a_types = [5, 9, 13, 36, 50]
    d_types = [1, 3, 30, 31]
    print('complet: ', df.shape)
    df_analog = df[df['ASDU_Type'].isin(a_types)]
    print('Choose analog: ', df_analog.shape)
    df_digital = df[df['ASDU_Type'].isin(d_types)]
    print('Choose digital', df_digital.shape)

    # Step 2: calculate features for each rolling window per id
    # 1. group by id
    # 2. for each id
    #     1. sort by time and observe time intervals
    #    2. divide indexes into windows
    #    3. use the previous windowed indexes index the subsample dataframes
    #    4. generate ts features for each subsample
    #    5. combine all subsample features
    # 3. combine all id and label by join
    total_t = df_analog['Time'].max() - df_analog['Time'].min()
    print('this dataset has time spanned: {} hours'.format(total_t / 3600))
    df_analog['datetime'] = df_analog.apply(lambda x : datetime.datetime.fromtimestamp(x['Time']), axis=1)
    df_analog['Day'] = df_analog.apply(lambda x : x['datetime'].day, axis=1)
    print(df_analog.head())

    # If test, pick the longest day as sample
    # Otherwise, extract TS features for each day, then concatenate all
    samples = pd.DataFrame()
    #for d in [4, 8, 9, 10, 11]:
    for d in [4, 8, 9, 10]:
        print('******* Day {} ********'.format(d))
        print(df_analog[df_analog['Day'] == d].shape)
        sample = df_analog[df_analog['Day'] == d].sort_values(by=['Time'])[['id', 'Time', 'Measurement']]
        total_t = sample['Time'].max() - sample['Time'].min()
        print('this dataset has time spanned: {} hours'.format(total_t / 3600))

        sample['DeltaT'] = sample.loc[:, 'Time'].diff().fillna(0)
        start_t1 = time.time()
        # 1. cumsum of deltaT
        sample['CumDeltaT'] = sample['DeltaT'].cumsum()
        # 2. divide by 60, name minute groups from division result by rounding
        n = 120
        sample['DivInterval'] = sample['CumDeltaT'].div(60 * n).round(0)
        print('Iteration takes {} seconds'.format(time.time() - start_t1))
        print('MAX value in DivInterval', sample['DivInterval'].max())
        sample_windows = sample.groupby(['DivInterval']).groups
        print('Divided into {} windows'.format(len(sample_windows)))
        #print(sample_windows[0])
        #sample.loc[sample_windows[0]]
        subsample_list = []
        for k in sample_windows.keys():
            subsample_list.append(sample.loc[sample_windows[k]][['id', 'Time', 'Measurement']])
        # params = ComprehensiveFCParameters()
        # del params['binnied_entropy']
        params = MinimalFCParameters()
        # params = EfficientFCParameters()

        sub_extracted = [my_extraction(sub, params) for sub in subsample_list]
        subsample_ts = pd.concat(sub_extracted)
        print('ts feature subsample_ts shape: ', subsample_ts.shape)
        samples = pd.concat([samples, subsample_ts])
        print('samples shape now: ', samples.shape)


    # Clean null and infinite values in samples
    # change the original 'rtu-ioa' indexes into a column
    print('Before handling null and infinite values, samples shape: ', samples.shape)
    samples.replace([np.inf, np.nan]).replace([-np.inf, np.nan])
    samples = samples.dropna(axis=1)
    samples = samples.reset_index().rename(columns={'index': 'ip-ioa'})
    # handle infinite values
    mask1 = (samples == np.inf)
    inf_cols = samples.loc[mask1.any(axis=1), mask1.any(axis=0)].columns
    samples.drop(inf_cols, axis=1, inplace=True)
    print('After handling null and infinite values, samples shape: ', samples.shape)
    # export
    out_df = samples
    out_fp = Path(out_p, 'bulk_samples_extracted_n={}_day48910.csv'.format(n))
    print('Output total size in subsample_list: ', out_df.shape, '\nExport to file: ', out_fp)
    out_df.to_csv(out_fp)


