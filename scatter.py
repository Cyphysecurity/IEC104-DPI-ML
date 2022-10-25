'''
This script creates template for scatter plots
input:
- any CSV
- manually change input file
- type through prompt input

'''

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv

#infile = input()                           # CSV file
df = pd.read_csv('apduRate_per_asduType_malformed.csv')     # manually change input
(x, y) = (df.iloc[:, 2], df.iloc[:, 5])
df1 = pd.read_csv('apduRate_all_good.csv')
(x1, y1) = (df1.iloc[:, 2], df1.iloc[:, 5])

fig = plt.figure()
fig, (ax1, ax2) = plt.subplots(ncols=2)
#plt.xlabel()
ax1.scatter(x, y, c = 'r', marker='x')
ax2.scatter(x1, y1, c = 'b', marker=',')
plt.grid()
plt.legend()
plt.show()