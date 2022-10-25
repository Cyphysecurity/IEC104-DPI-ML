from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib.ticker import FuncFormatter, MaxNLocator
import sys
import re
import csv
pl.rcParams["backend"] = "TkAgg"



def preprocessdata(dPath, yAxisLable, title):
    
    '''
    with open(file) as f:
        plots = csv.reader(f, delimiter=':')
        for row in plots:
            y.append(int(row[0]))
            x.append(float(row[1]))
    Note: Either method works the same, but only one line when using numpy
    '''
    files = [join(dPath, file) for file in listdir(dPath) if isfile(join(dPath, file))]

    for file in files:
        x = []
        y = []
        filename = file.replace("txt", "pcap").replace("input/", "")
        y, x = np.loadtxt(file, delimiter=':', unpack=True)
        pl.plot(x, y, label=filename)
    pl.xlabel("Time in seconds")
    pl.ylabel(yAxisLable)
    pl.title(title)
    pl.legend()
    pl.grid(True)
    pl.show()

def cleanString(line):
    str = line.strip().replace(' ', '')
    #print(str)
    return (str.replace('\n', ''))

if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print("Missing input file: \n")
    else:
        datapath = sys.argv[1]
        yLabel = sys.argv[2]
        title = sys.argv[3]
        preprocessdata(datapath, yLabel, title)