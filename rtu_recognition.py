import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pathlib, os
from pathlib import Path, WindowsPath
import csv, re
from collections import OrderedDict

hist_param = {
    'x': [],
    'bins': 0,
    'histtype': 'step',
    'hatch': '/',
    'density': False
}

_servers = ['172.31.1.100', '172.31.1.102']
_known_rtu = [22,    26,    27,    28,    29,    37,    48,    50,    72,
          73,    92,    93,    96,    99,   172,   184,   221,   228,
         230, 11027]

def myhist(ax, hist_param):
    '''
    Make histogram graphs
    Output:
    (x, y, z): x -- the counts; y -- the edges; z -- plot related
    '''
    return ax.hist(hist_param['x'], bins=hist_param['bins'], histtype=hist_param['histtype'], hatch=hist_param['hatch'], density=hist_param['density'])
def hist_fv(data, edges):
    '''
    Transform into histogram counts as feature vector
    Kw arguments:
    data -- 1-D raw data
    edges -- setting 1: one integer, the number of buckets; setting 2: a list of numbers, the edges/ticks of the buckets
    Output:
    (x, y): x -- the counts; y -- bin edges
    '''
    return np.histogram(data, bins=edges)

#def preprocess():

#def get_hist(df):


#def run_knn(x):


if __name__ == '__main__':
    # Essential universal paths here
    cur_p = Path()
    home_p = cur_p.home()
    out_p = Path(cur_p.absolute().parents[0], 'output')
    print('current directory: {} \nhome directory: {}\noutput: {} \n'.format(cur_p.absolute(), home_p, out_p.absolute()))
    known_p = Path('D:', 'Google Drive', 'IEC104', 'Netherlands-results', 'gas_dataset_point_configuration_recolumned.csv')
    known_df = pd.read_csv(known_p, sep=',')
    unknown_p = Path()
    unknown_df = pd.read_csv(unknown_p, sep=',')
    print('known RTU data: ', known_df.columns, known_df.shape, \
          'unknown RTU data: ', unknown_df.columns, unknown_df.shape)
