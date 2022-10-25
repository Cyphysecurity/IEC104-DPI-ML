'''
Created by xxq160330 at 5/24/2018 7:10 PM
This script is the top file.
Call other modules as needed
Also test small piece of code
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import stats

from sklearn import decomposition, preprocessing
from sklearn import datasets
from sklearn.cluster import KMeans
import featureSelection
import csv

if __name__ == '__main__':
    print('main.py running')
