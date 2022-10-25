'''
Created by xxq160330 at 9/4/2018 2:02 PM
This script is a trial to apply ARMA on time series data
'''

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

import matplotlib.pyplot as plt
import src.rtu.utilities as utilities

# loop through all RTUs under each type
# plot TS for each
def ts_plot(infile, types):
    for type in types:
        df = infile.loc[infile['ASDU_Type'] == type]
        ioas = df.IOA.unique()
        for ioa in ioas:
            dft = df.loc[df['IOA'] == ioa]
            x0 = dft['Time']
            x = x0.apply(utilities.epochConverter)
            y = dft['Measurement']
            l = [x.min(), x.max()]
            fig, ax = plt.subplots(1, 1)
            ax.plot(x, y)
            ax.set_title('Type %d IOA %d' % (type, ioa))
            ax.set_xticks([x.iloc[0], x.iloc[(len(x) - 1)]])
            ax.set_xticklabels(l)
            fig.show()

# plot stationarity features for a single time series
# results in time series values, autocorrelation and normality plots
def stationarity(y):
    figsize = (15, 10)
    lags = 10
    fig = plt.figure(figsize=figsize)
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    pp_ax = plt.subplot2grid(layout, (2, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title('Time Series Analysis Plots')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')
    scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

    plt.tight_layout()
    plt.savefig('stationarity.svg')
    plt.show()
    print('mean: %.3f\nvariance: %.3f\nstandard deviation: %.3f\n' % (y.mean(), y.var(), y.std()))

if __name__ == '__main__':
    infile = pd.read_csv('D:\PycharmProjects\XM\data\\192.168.111.24.csv', delimiter=',')
    types = infile.ASDU_Type.unique()

    df = infile.loc[infile['ASDU_Type'] == 13]
    df1006 = df.loc[df['IOA'] == 1006]
    y = df1006['Measurement']
    stationarity(y)
    #ts_plot(infile, types)