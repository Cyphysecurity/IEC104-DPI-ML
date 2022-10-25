'''
Created by grace at 5/13/18 4:56 PM
This script will generate sample data
'''

import csv
import numpy as np

data = np.array([['2', '1-1001', '50', '20', '23'], ['3', '1-10', '50', '60', '63']])
with open('sample.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['item','system','asdu_type','min','max'])
    filewriter.writerow(['1', '2-1000','36','1','10'])
    filewriter.writerow(['2', '1-1001', '50', '20', '23'])
    filewriter.writerow(['3', '1-10', '50', '60', '63'])

csvfile.close()