#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib.dates import epoch2num
import src.rtu.labeling as labeling
import src.rtu.utilities as utilities
import src.rtu.preprocessing as prep
import src.rtu.learning as learning
import src.rtu.plotting as plotting
import src.rtu.other_stats as ostats

import time
import sys
from os import listdir
from os import path


# In[2]:


# INPUT file to learn, generated from choice 4


# In[14]:


df = pd.read_csv('D:\PycharmProjects\XM\prepped\stats\stats_combined.csv', delimiter=',')
dft = prep.filterLabel(df)
dft = prep.mergeCurPow(dft)
dft['categorical_id'] = dft['Physical_Type'].factorize()[0]  # numeric categorical ID
features = ['mean', 'min', 'max', 'var', 'count', 'autocorr']
print('Data before shuffling:')
print('Verify columns in X: \n', dft[features].head(2))
print('Verify column in y: \n', dft['categorical_id'].iloc[:2])


# In[15]:


print('Data after shuffling:')
dft = dft.reindex(np.random.permutation(dft.index)).sort_index()
print('Verify columns in X: \n', dft[features].head(2))
print('Verify column in y: \n', dft['categorical_id'].iloc[:2])


# ### Plot imbalanced classes

# In[5]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6))
dft.groupby(by='Physical_Type')['mean'].count().plot.bar()

plt.xlabel('Physical Types')
plt.ylabel('Count of Data Samples')
plt.tight_layout()
plt.savefig('imbalanced_class_merged_phylabels.svg')
plt.show()


# ### Benchmark algorithms I: with Min-Max Scalor

# In[16]:


# Generate features, labels
# scaling or not, run each learning algorithm
features = dft[features].astype(float).values
features_scaled = prep.scaling(features)
labels = dft['categorical_id'].values
print('feature vector dimension is %s' % str(features.shape))
print('label dimension is %s' % str(labels.shape))
features_scaled


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


models = [
    DecisionTreeClassifier(random_state=10),
    KNeighborsClassifier(n_neighbors=3),
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    #MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 10
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
i = 0
while i < 100: 
    for model in models:
      model_name = model.__class__.__name__
      accuracies = cross_val_score(model, features_scaled, labels, scoring='accuracy', cv=CV)
      for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    i += 1


# In[ ]:


import seaborn as sns

ax = sns.boxplot(x='model_name', y='accuracy', data=cv_df)
ax = sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.xticks(rotation=30)
plt.title('Feature Vector Scaled in Min-Max Scaler')
plt.savefig('accuracy_allmodels_merged_phylabels.svg')
plt.show()


# In[ ]:


print('There are %d models under %d folds in %d iterations. cv_df shape is %s' % (len(models), CV, i, str(cv_df.shape)))
cv_df.head(10)


# ### Benchmark algorithms II: without scaling, preserving the negative values

# In[8]:


# Apply Stratified KFold as well
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=CV)
skf.get_n_splits(features, labels)
cv_df2 = pd.DataFrame(index=range(CV * len(models)))
entries2 = []
fold_idx2 = 0
for train_index, test_index in skf.split(features, labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    #accuracy_allms = []
    for model in models:
        model_name = model.__class__.__name__
        print(model_name)
        clf = model.fit(X_train, y_train)
        accuracy2 = clf.score(X_test, y_test)
        entries.append((model_name, fold_idx2, accuracy2))
        #accuracy_allms.append(accuracy)
    fold_idx2 += 1
cv_df2 = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# In[ ]:


# plot accuracy per model after cross validation
ax1 = sns.boxplot(x='model_name', y='accuracy', data=cv_df)
ax1 = sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.xticks(rotation=30)
plt.title('Feature Vector Not Scaled')
plt.show()


# ### Focus on model with the highest accuracy
# ### Model evaluation in confusion matrix

# In[17]:


import seaborn as sns
from sklearn.model_selection import train_test_split
bestmodel = models[0]
splitperc = 0.3
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, dft.index, test_size=splitperc, random_state=0)
bestmodel.fit(X_train, y_train)
y_pred = bestmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)
#labels = ['Power', 'Q', 'U', 'I', 'Status', 'Frequ', 'AGC-SP (Set-Point)', 'I-three', 'TapPosMv']
ticklabels = dft.Physical_Type.unique()
#labels = np.insert(labels, 0, '0')
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=ticklabels, yticklabels=ticklabels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Test set size is {}'.format(splitperc))
plt.savefig('confusionM_merged_phylabels.svg')
plt.show()


# ### Math form of confusion matrix 

# In[18]:


conf_mat


# ### Display misclassifications 

# In[84]:


categorical_id_df = dft[['Physical_Type', 'categorical_id']].drop_duplicates().sort_values('categorical_id')
#phy_to_id = dict(categorical_id_df.values)    # a dict, key = categ, value = id
id_to_phy = dict(categorical_id_df[['categorical_id', 'Physical_Type']].values)           # a dict, key = id, value = categ
id_to_phy


# ### Print out all the wrongly classified data samples 

# In[90]:


def isequal(a):
    if a[0] == a[1]:
        return True
    else:
        return False
combined = np.stack((y_test, y_pred), axis=-1)   
falselist = np.apply_along_axis(isequal, 1, combined)  # a list recording boolean values, when predicted is equal to actual or not
testindexlist = []
for i in np.arange(len(falselist)):
    if falselist[i] == False:
        testindexlist.append(indices_test[i])
indices_test1 = pd.Index(testindexlist)     # get index for wrongly classified samples

dft1 = dft.loc[indices_test1]     # data frame with only wrongly classfied samples
wrongy_pred = y_pred[[i for i in np.arange(len(falselist)) if falselist[i] == False]]
wrongphy_pred = []
for id in wrongy_pred:
    wrongphy_pred.append(id_to_phy[id])
dft1['Predicted_Phy'] = pd.Series(wrongphy_pred, index=indices_test1)
from IPython.display import display
display(dft1.loc[indices_test1][['Signature', 'mean', 'min', 'max', 'var', 'count', 'autocorr', 'Physical_Type', 'Predicted_Phy']])
#display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Product', 'Consumer_complaint_narrative']])


# ### Find the terms that are the most correlated with each category with chi-squared test 

# In[ ]:





# In[ ]:





# ### Report classification for each class

# In[81]:


from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=dft['Physical_Type'].unique()))


# In[ ]:




