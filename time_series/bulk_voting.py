'''
For each point variable
Take majority voting for classification result
'''

import pandas as pd
import numpy as np
import pathlib, time, sys, re
from pathlib import Path, WindowsPath
import matplotlib.pyplot as plt
import seaborn as sb

if __name__ == '__main__':
    cur_p = Path().absolute()
    #in_p = Path('G:\My Drive', 'IEC104', 'Netherlands-results', 'Ioa_timeseries', 'rtus-22-26-172-228-230-11024-11027_alltime')  # 'known_rtus-22-26-92-172_alltime') 'known_rtus_rest_alltime')  # '04-15')
    in_p = Path(Path().absolute().parents[1], 'output')
    out_p = in_p
    print('Input path: {}\nOutput path: {}\n'.format(in_p, out_p))

    pred_f = Path(in_p, 'bulk_test_w_pred.csv')

    pred_df = pd.read_csv(pred_f)
    result = pd.DataFrame(columns=['ip-ioa', 'Label', 'CorrectCnt', 'CorrectPercent']) # header = ip-ioa/id, label, correct#, correct%

    # step 1: group by point
    groups = pred_df.groupby(by=['rtu-ioa']).groups
    print('There are {} unique points'.format(len(groups.keys())))
    # step 2: for each point, count # and % of correct predictions --> majority vote
    for id in groups.keys():
        cur_df = pred_df.loc[groups[id]]
        cnt = cur_df[cur_df['Label'] == cur_df['Pred']].shape[0]
        percent = np.divide(cnt, cur_df.shape[0])
        result = result.append({'rtu-ioa': id, 'Label': cur_df['Label'].unique()[0], 'CorrectCnt': cnt, 'CorrectPercent': percent}, ignore_index=True)
    print(result)
    print('Correct percentage unique values: ', result['CorrectPercent'].unique())
    for p in [0.3, 0.5, 0.8, 0.9]:
        print('Correct percentage > {}% = {}'.format(p * 100, result[result['CorrectPercent'] > p].shape[0]))
    for wp in [0.3, 0.5, 0.8, 0.9]:
        wrong_result = result[result['CorrectPercent'] < wp]
        print('Total classification class distribution:\n{}, \nWrong classification threshold: {}, distribution: \n{}'.format(result['Label'].value_counts(), wp, wrong_result['Label'].value_counts()))



