'''
Created by xxq160330 at 10/17/2018 8:54 PM
Hold any utility functions here
'''
import pandas as pd
import numpy as np
import time

# convert epoch timetag into formatted dates
def epochConverter(t):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
