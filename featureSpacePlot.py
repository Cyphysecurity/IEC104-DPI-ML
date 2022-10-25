'''
Created by xxq160330 at 8/20/2018 12:36 PM
'''
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('k=5_pca_label.csv')
fs = pd.concat([df.iloc[:, 3], df.iloc[:, 6], df.iloc[:, 8:10], df.iloc[:, 18]], axis=1)

#colors = ['navy', 'tomato', 'turquoise', 'darkorange', 'red', 'orange', 'plum', 'grey', 'olive', 'brown']
#markers = ['^', 's', 'o', 'd', 'x', '1', '2', '3', '4', '*']
plt.figure(figsize=(5, 4))
plt.subplot(4, 1, 1)
#for color, i, label in zip(colors[0:4], range(0,4), markers[0:4]):
#    for j in fs.index:
#        plt.plot(j, fs.iloc[j, 0], color=color, marker=label)
plt.plot(fs.index, fs.iloc[:, 0], 'bo-', label='Direction', markersize=2)
plt.yticks([0,1])
plt.legend()
plt.subplot(4, 1, 2)
#for color, i, label in zip(colors[0:4], range(0,4), markers[0:4]):
#    for j in fs.index:
#        plt.plot(j, fs.iloc[j, 1], color=color, marker=label, markersize=2)
plt.plot(fs.index, fs.iloc[:, 1], 'bo-', label='Average time interval', markersize=2)
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(fs.index, fs.iloc[:, 2], 'bo-', label='Total payload size', markersize=2)
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(fs.index, fs.iloc[:, 3], 'bo-', label='Number of packets', markersize=2)
plt.legend()
plt.tight_layout()
plt.savefig('featureSpace_4in1.svg')
plt.show()