'''
Created by xxq160330 at 5/14/2018 5:31 PM
This script plots parallel coordinates
Each attribute will be placed on a single axis and have its own scale
'''
import pandas as pd
from pandas.plotting import parallel_coordinates
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


data = pd.read_csv('pktRate_vs_Time.csv')
print('finish loading data...')

cols = ['Time', 'Pkt_rate']
x = [i for i, _ in enumerate(cols)]
fig, axes = plt.subplots(1, len(x) - 1, sharey=False)

# normalize for each attribute
min_max_range = {}
for col in cols:
    min_max_range[col] = [data[col].min(), data[col].max(), np.ptp(data[col])]
    data[col] = np.true_divide(data[col] - data[col].min(), np.ptp(data[col]))



plt.figure()
print('start plotting...')
parallel_coordinates(data, 'label')
print('finish!!!')
plt.show()
