from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as pl
import sys
import re
import csv
'''
This Python script will plot a pie chart the the supplied
data which come from files that are located in the folder that is provided
are input argument on the command line. The input files should all have the
following format <Key:Value>, where Key is the name or label of the data to be plotted
and Value is the unit count for this specific label. Below show an example of how
the input file should contain:
I-FORMAT:74
S-FORMAT:11
U-FORMAT:3

Key = I-FORMAT, S-FORMAT, U-FORMAT which are the labels of the pie chart slices
Value = 74, 11, 3 which are the unit count that corresponds to its own label 
i.e. I-FORMAT's unit count is 74, S and U each has 11 and 3 respectively

Based on these information, I-FORMAT has 84.1% while S and U each has 12.5% and 3.4% respectively  

'''

def PlotPieChart(inputDir, pieTitle):
    
    files = [join(inputDir, file) for file in listdir(inputDir) if isfile(join(inputDir, file))]
    isLabelDone = False
    value = 0
    labels = []
    slices = []
    for file in files:
        with open(file) as f:
            index = 0
            plots = csv.reader(f, delimiter=':')
            for row in plots:
                
                # Each file has 3 row with the following <Key:Value> example:
                #I-FORMAT:74,  in this case Key=I-FORMAT which is used as label for the Pie chart
                #S-FORMAT:11,  and Value= 74 or 11, or 3 and is used for slice % of the Pie chart
                #U-FORMAT:3
                # Since we only need to get labels once, from the first file
                # Rest of the files, we just need to add up the values for each row
                if (isLabelDone == False):
                    labels.append(str(row[0]))
                    slices.append(int(row[1]))
                else:
                # Rest of the files, just add up the current vallue to the latest one 
                    value = slices[index] + int(row[1])
                    slices[index] = value
                index += 1            
            isLabelDone = True
    # Now start ploting        
    fig = pl.figure(1, figsize=(8,8))
    pl.pie(slices, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    pl.title(pieTitle, bbox={'facecolor':'0.8', 'pad':5})
    pl.show()

# Method to perform string cleaning, sanitizing
def cleanString(line):
    str = line.strip().replace(' ', '')
    #print(str)
    return (str.replace('\n', ''))

# Main method
# Usage: python PlotPieFormatypes.py <inputFolderName/> "Title of Pie Chart"
if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("Missing input command line parameters!")
        print("Usage: python PlotPieFormatypes.py <inputFolderName/> \"Title of Pie Chart\"")
        print("where <inputFolderName> is were all the files that contains plotting information located")
    else:
        inputDir = sys.argv[1]
        pieTitle = sys.argv[2]
        PlotPieChart(inputDir, pieTitle)