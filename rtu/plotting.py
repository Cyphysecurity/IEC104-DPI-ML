'''
Created by xxq160330 at 9/19/2018 4:24 PM
This script keeps all plotting methods
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import epoch2num
from matplotlib.ticker import Formatter

import src.rtu.utilities as utilities

# X = time data in x-axis, y = measurement data in y-axis
# phyType is plotting for each physical type
# time series scatter plot and measurement box plot
def ts_plot(X, y, phyType):
    x0 = X.apply(utilities.epochConverter)
    l = [x0.min(), x0.max()]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.scatter(x0, y, s=5, alpha=0.7)
    ax1.set_title('Time series for %s type measurement' % phyType)
    ax1.set_xticks([x0.iloc[0], x0.iloc[(len(x0) - 1)]])
    ax1.set_xticklabels(l)
    ax2.boxplot(y)
    figname = phyType + '_ts.png'
    fig.savefig(figname)
    fig.show()
    print('******** plotting for %s finishes!!! *********' % phyType)

# bar plot showing stats metrics, old version
def statsBar1(labels, rows):
    fig, ax = plt.subplots()
    bar_width = 0.1
    opacity = 0.7
    index = np.arange(4)
    rect = np.arange(1, 9, 1)
    colors = ['turquoise', 'navy', 'plum', 'grey', 'olive', 'brown', 'tomato',  'darkorange', 'red', 'orange']

    r = np.arange(9)
    for i in rect:
        bar_start = index + (i - 1) * bar_width
        #r[i] = ax.bar(bar_start, rows[i], bar_width, alpha=opacity, color=colors[i], label=labels[i])
        ax.bar(bar_start, rows[i][:5], bar_width, alpha=opacity, color=colors[i], label=labels[i])

    ax.set_xlabel('Statistical metrics')
    ax.set_ylabel('Value')
    ax.set_xticks(index)
    ax.set_xticklabels(['mean', 'min', 'max', 'std', 'count', 'autocorr'])
    ax.legend()

    fig.tight_layout()
    plt.savefig('stats.eps')
    plt.show()

# Plot any distribution with pie chart
def pieDistri(data):
    legends = ['M1', 'M2', 'M3', 'REST']
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    recipe = ["225 g flour",
              "90 g sugar",
              "1 egg",
              "60 g butter",
              "100 ml milk",
              "1/2 package of yeast"]

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(legends[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title('ASDU Types')
    plt.show()

# AGC
def plotagc(dfList):
    colors = ['black', 'yellow', 'turquoise', 'red', 'orchid', 'pink', 'purple', 'salmon', 'navy', 'sienna', 'plum', 'grey', 'olive', 'brown', 'tomato', 'darkorange', 'orange', 'green', 'maroon']
    symbols = ['bP', 'r>', 'yx', 'g.', ]
    print('****** Plotting AGC right now... **********')
    fig, axes = plt.subplots(figsize=(30, 8))
    plt.axis('off')     # not showing main plot's axis
    #ax = fig.add_subplot(1,1,1)
    for day, d in zip(dfList, np.arange(len(dfList))):
        startT = np.min(day.Time)
        print('Now is plotting srcIP %s on day %s' % (str(day.srcIP.unique()), utilities.epochConverter(startT)))
        ax = fig.add_subplot(1, len(dfList), d + 1)
        groupedp = day.groupby(by='IOA')
        ioas = groupedp.groups.keys()
        labels = ['ioa' + (str(k) + str(groupedp.get_group(k).Physical_Type.unique())) for k in groupedp.groups.keys()]
        index = np.arange(len(groupedp.groups.keys()))

        #for ioa, s, l, i in zip(ioas, symbols, labels, index):
        for ioa, l, i, c in zip(ioas, labels, index, colors):
            print('IOA %s' % ioa)
            #ax.plot(groupedp.get_group(ioa)['Time'], groupedp.get_group(ioa)['Measurement'], s, markersize=5, label=l)
            ax.plot(groupedp.get_group(ioa)['Time'], groupedp.get_group(ioa)['Measurement'], color=c, marker='o', markersize=5, label=l)
        ax.set_xticks([np.min(day['Time']), np.max(day['Time'])])
        ax.set_xticklabels([pd.to_datetime(np.min(day['Time']), unit='s').tz_localize('UTC').tz_convert('US/Central'),
                            pd.to_datetime(np.max(day['Time']), unit='s').tz_localize('UTC').tz_convert('US/Central')], rotation=15)
        if (len(day.srcIP.unique()) == 1) and (len(day.dstIP.unique()) == 1):
            ax.set_title(day.srcIP.unique()[0] + '->' + day.dstIP.unique()[0])
        else:
            ax.set_title(str(day.srcIP.unique()) + '->' + day.dstIP.unique())
        ax.legend()
        ax.grid()

    plt.tight_layout(h_pad=0, w_pad=0)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('agc_alldays.png')
    plt.show()

# plot Time Interval in AGC
def plot_deltat(deltaT, arr_t, cntApdu):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # ax1: CDF of APDU inter-arrival time
    ax1.plot(arr_t, cntApdu)
    # ax2: boxplot of time intervals
    ax2.boxplot(deltaT)
    ax2.set_yticks([np.min(deltaT), np.max(deltaT)])
    #ax.set_yticklabels()
    ax2.set_title('Time intervals ')

def plot_2dhist(X, Y):
    plt.hist2d(X, Y, bins=(50, 50), cmap='Greys_r')
    plt.colorbar()
    plt.savefig('2dhist.png')
    plt.show()
