'''
Created by grace at 8/10/18 8:12 PM
This script uses heatmap in Seaborn
''' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



y = np.random.randint(1,100,40)
y = y.reshape((5,8))
df = pd.DataFrame(y,columns=[x for x in 'abcdefgh'])

sns.heatmap(df)     # plot data frame or array
plt.show()
