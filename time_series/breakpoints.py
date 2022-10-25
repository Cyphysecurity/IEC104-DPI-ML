'''
This script applies breakpoints/segments analysis
One way of anomaly detection in time series
Tools: Ruptures
'''

import ruptures as rpt
import pandas as pd
import numpy as np
import scipy.stats as st
import pathlib, time, datetime, sys
from pathlib import Path, WindowsPath
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq


def apply_model(x: np.ndarray, n_break: int, submodel: str):
    """
    Model initiates, trains and predicts
    Parameters:
        x: 1-D data
        n_break: number of breakpoints
    Returns:
        indexes of break points
    """
    print('****** Start breakpoint-search *******')

    #rpt.Dynp(model=submodel), rpt.Pelt(model=submodel), rpt.KernelCPD(kernel="linear"), rpt.BottomUp(model=submodel), rpt.Window(model=submodel)
    model = rpt.Binseg(model=submodel)
    #model = rpt.Dynp(model=submodel)
    start_t = time.time()
    model.fit(x)
    print('Model {} training takes {} minutes'.format(model, (time.time() - start_t) / 60))
    n_break -= 1
    print('Breakpoints search takes long time, please be patient...')
    bp_idx = model.predict(n_bkps=n_break)
    print('Breakpoints searching takes {} minutes'.format((time.time() - start_t) / 60))

    print(type(bp_idx), len(bp_idx))
    return bp_idx, model


def plot_seg(signal, bp_idx):
    """
    Plot the time series with segmentation
    """
    bkps = signal[bp_idx]
    rpt.show.display(signal, bkps, bp_idx, figsize=(10, 6))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    plt.show()
    return bkps


def get_ci(x, bp_idx, p):
    """
    Compute CI (confidence intervals)
    Parameters:
        x: data in numpy array
        bp_idx: breakpoint indexes in a list
        p: alpha value, e.g. 0.05
    """
    # create 95% confidence interval for population mean weight
    intervals = []
    start = 0
    for i in range(len(bp_idx)):
        end = bp_idx[i]
        if i == len(bp_idx) - 1:
            cur_x = x[start:]
        else:
            cur_x = x[start:end]
            start = end

        var = np.var(cur_x)
        mean = np.mean(cur_x)
        n = len(cur_x)
        degree_fredom = n - 1
        #t = st.t.ppf(1 - p / 2, degree_fredom)  # two-tail t-critical value for 95%
        #s = np.std(cur_x)  # sample standard deviation
        #lower = np.mean(cur_x) - (t * s / np.sqrt(n)) # T-test
        #upper = np.mean(cur_x) + (t * s / np.sqrt(n))
        #lower = mean - degree_fredom * var / st.chi2.ppf(1 - p / 2, degree_fredom)   # chi-square test
        #upper = mean + degree_fredom * var / st.chi2.ppf(p / 2, degree_fredom)

        #lower = 0
        #upper = var*(st.chi2.isf(q=p / 2, df=degree_fredom) / degree_fredom)


        lower = mean - var
        upper = mean + var
        cur_int = (lower, upper)
        #cur_int = st.norm.interval(alpha=p, loc=np.mean(cur_x), scale=st.sem(cur_x))
        #cur_int = st.t.interval(alpha=p, df=degree_fredom, loc=np.mean(cur_x), scale=st.sem(cur_x))
        #cur_int = (np.percentile(cur_x, 0.05), np.percentile(cur_x, 0.95))
        intervals.append(cur_int)
        print('min = {}, max = {}, mean = {}'.format(np.min(cur_x), np.max(cur_x), np.mean(cur_x)))
        #from statsmodels.stats.weightstats import ztest as ztest
        #zscore, pval = ztest(cur_x, value=100)
        #print('z-score = {}, p value = {}, mean = {}'.format(zscore, pval, np.mean(cur_x)))
    return intervals


def get_anomaly(x, bp_idx, intervals):
    """
    Locate outliers based on confidence intervals
    Returns:
        anomalies: a list of anomaly indexes in each interval
    """
    #x = rfftfreq(len(x), 1)
    anomalies = []
    start = 0
    for i in range(len(bp_idx)):
        low, high = intervals[i]
        end = bp_idx[i]
        print('i = {}, start = {}, end = {}'.format(i, start, end))
        if i == len(bp_idx) - 1:
            if low is None or high is None:
                continue
            cur_x = x[start:]
        else:
            cur_x = x[start:end]

        anomaly_idx = np.where(np.logical_or(cur_x < low, cur_x > high))
        print('This interval has {} anomalies'.format(len(anomaly_idx[0])))
        anomalies += list(np.add(anomaly_idx[0], start))
        start = end
        #anomalies += anomaly_idx[0].tolist()
    return anomalies


def plot_anomaly(x, anomalies):
    colorMap = []
    plt.plot()
    for i in range(len(x)):
    #for i in range(12794, 16082, 1):
        if i in anomalies:
            colorMap.append('r')
        else:
            colorMap.append('b')

    plt.scatter([i for i in range(len(x))], x, c=colorMap, s=6)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.show()


def compute_anomly_score(x, bp_idx, interval):
    """
    Compute the anomly score
    """

    return


