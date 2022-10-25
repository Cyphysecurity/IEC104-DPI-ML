'''
## This script:
1. get all input files parsed from 104 parser
2. TSFRESH feature engineering
'''
import tsfresh
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
import pandas as pd
import numpy as np
import pathlib, time, sys, re
from pathlib import Path, WindowsPath
import matplotlib.pyplot as plt
import seaborn as sb
from utility import *


def extract_tsfresh(df, out_p):
    '''
    # Feature extraction step: calculate features for each rolling window per id
    # 1. group by id
    # 2. for each id
    #     1. sort by time and observe time intervals
    #    2. divide indexes into windows
    #    3. use the previous windowed indexes index the subsample dataframes
    #    4. generate ts features for each subsample
    #    5. combine all subsample features
    # If test, pick the longest day as sample
    # Otherwise, extract TS features for each day, then concatenate all
    '''
    samples = pd.DataFrame()
    sample = df.sort_values(by=['Time'])[['id', 'Time', 'Measurement']]
    total_t = sample['Time'].max() - sample['Time'].min()
    print('this dataset has shape: {}, time spanned: {} hours'.format(sample.shape, total_t / 3600))

    sample['DeltaT'] = sample.loc[:, 'Time'].diff().fillna(0)
    start_t1 = time.time()
    # 1. cumsum of deltaT
    sample['CumDeltaT'] = sample['DeltaT'].cumsum()
    # 2. divide by 60*n, name minute groups from division result by rounding
    n = 720
    sample['DivInterval'] = sample['CumDeltaT'].div(60 * n).round(0)
    print('Iteration takes {} seconds'.format(time.time() - start_t1))
    print('MAX value in DivInterval', sample['DivInterval'].max())
    sample_windows = sample.groupby(['DivInterval']).groups
    print('Divided into {} windows'.format(len(sample_windows)))
    #print(sample_windows[0])
    #sample.loc[sample_windows[0]]
    subsample_list = []
    start_t2 = time.time()
    sub_extracted = []
    params = ComprehensiveFCParameters()
    print('There are {} keys in params'.format(len(params.keys())))
    print('\n*** Start tsfresh extraction ***')
    for k in sample_windows.keys():
        if k % 100 == 0:
            print('************ Window ', k, ' *************')
        #subsample_list.append()
        sub = sample.loc[sample_windows[k]][['id', 'Time', 'Measurement']]
        sub_extracted.append(my_extraction(sub, params))


    # params = MinimalFCParameters()
    # params = EfficientFCParameters()
    # del params['binned_entropy']  # if any parameter invalid during calculation, delete
    # sub_extracted = [my_extraction(sub, params) for sub in subsample_list]
    print('TSFRESH takes {} minutes'.format((time.time() - start_t2) / 60))
    subsample_ts = pd.concat(sub_extracted)
    print('ts feature subsample_ts shape: ', subsample_ts.shape)
    samples = pd.concat([samples, subsample_ts])
    print('samples shape now: ', samples.shape)


    # Clean null and infinite values in samples
    # change the original 'rtu-ioa' indexes into a column
    print('Before handling null and infinite values, samples shape: ', samples.shape)
    samples.replace([np.inf, np.nan]).replace([-np.inf, np.nan])
    samples = samples.dropna(axis=1)
    samples = samples.reset_index().rename(columns={'index': 'ip-ioa'}) # TODO: unify 'rtu-ioa', 'ip-ioa'
    # handle infinite values
    mask1 = (samples == np.inf)
    inf_cols = samples.loc[mask1.any(axis=1), mask1.any(axis=0)].columns
    samples.drop(inf_cols, axis=1, inplace=True)
    print('After handling null and infinite values, samples shape: ', samples.shape)
    # export
    out_df = samples
    out_fp = Path(out_p, 'gas_samples_extracted_n={}.csv'.format(n))
    # TODO: output folder customized by running timestamp
    print('Output total size in subsample_list: ', out_df.shape, '\nExport to file: \n', out_fp)
    out_df.to_csv(out_fp)


def my_extraction(df, params):
    return extract_features(df, column_id='id', column_sort='Time', default_fc_parameters=params)
