'''
Created by grace at 7/31/18 10:42 PM
Draw stem plot from input file
''' 

from matplotlib import axes
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# input file to plot
infile = pd.read_csv('/home/grace/PycharmProjects/iccp/combinedProcessed.csv')
dfs = pd.concat([infile.iloc[:, 2], infile.iloc[:, 5], infile.iloc[:, 7:9]], axis=1)
m = dfs.as_matrix()
mt = m.transpose()
#dir = mt[0]
#avgT = mt[1]
#num = mt[2]
#apduLen = mt[3]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
xrange = np.linspace(0, 9, 10)
cmap = ['r', 'b', 'g', 'y']
xticks = ['C3-O10', 'C1-O2', 'C1-O35', 'C2-O20', 'C1-O38', 'O28-C1', 'C3-O12', 'C1-O37', 'C2-O24', 'C3-O13']

for color, i in zip(cmap, [0, 1, 2, 3]):
    lfmt = cmap[i] + '-'
    bfmt = cmap[i] + '-'
    markerline, stemlines, baseline = axes[i].stem(xrange, mt[i][:10], linefmt=lfmt, basefmt=bfmt)

    axes[i].set_xticks(xrange)
    axes[i].set_xticklabels(xticks)
    axes[i].tick_params(axis='x', labelrotation=45)
    if (i == 0):
        axes[i].set_yticks([0, 1])
        axes[i].set_title('Direction of Flow \n(unit: 1 = from central stations to outstations\n0 = from outstations to central stations)')
    elif (i == 1):
        axes[i].set_title('Average Time Interval\n(unit: seconds)')
    elif (i == 2):
        axes[i].set_title('Total Packet Count')
    elif (i == 3):
        axes[i].set_title('Total Payload Size\n(unit: number of octets)')


plt.savefig('stemPlot.svg')
plt.show()
