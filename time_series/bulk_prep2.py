'''
## This script runs for BULK:
1. categorical feature engineering
2. add labels

## Supervised learning pipeline
Step 1: bulk_parse_device_list.py -> generate tidied point labels
Step 2: bulk_prep.py -> generate TSFRESH features, clean null & infinite, and export for storage
Step 3: bulk_prep2.py -> combine labels, prepare X, y, temporary holder for classifiers
'''

import tsfresh
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import numpy as np
import pathlib, time, sys, re, datetime
from pathlib import Path, WindowsPath
import matplotlib.pyplot as plt
import seaborn as sb
import sktime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters

# Utility functions

if __name__ == '__main__':
    # I/O
    cur_p = Path().absolute()
    # G:\My Drive\IEC104\XM-results\IOA_timeseries\bulk2017
    in_p = Path('G:\My Drive', 'IEC104', 'XM-results', 'IOA_timeseries', 'bulk2017')
    out_p = Path(Path().absolute().parents[1], 'output')
    print('Current path: {}\nInput path: {}\nOutput path: {}'.format(cur_p, in_p, out_p))
    samples = pd.read_csv(Path(out_p, 'bulk_samples_extracted_n=60_day48910.csv'))
    #samples = pd.read_csv(Path(out_p, 'bulk_samples_extracted_n=1.csv')).drop(['Unnamed: 0', 'rtu-ioa'], axis=1)
    if 'Unnamed: 0.1' in samples.columns:
        samples.rename({'Unnamed: 0.1': 'ip-ioa'}, axis=1, inplace=True)
    print('samples: ', samples.shape)
    # categorical features
    print('\n******* Start process categorical features *******')
    df = pd.DataFrame() # total time series original df
    cnt = 0
    if not in_p.exists():
        print('Google Drive may not be active!!!')
        sys.exit()
    for f in in_p.glob('*'):
        # print(f)
        cur_df = pd.read_csv(f)
        cnt += cur_df.shape[0]
        df = pd.concat([df, cur_df])
    print('After concatenation, data row # equal to summing up all the individual dataframes? ', cnt == df.shape[0],
          df.shape[0])
    # Simple prep
    df['ip-ioa'] = df['srcIP'].astype(str) + '-' + df['IOA'].astype(str)
    all_categorical = pd.merge(samples['ip-ioa'], df[['ip-ioa', 'ASDU_Type', 'CauseTx']].drop_duplicates(subset=['ip-ioa']),
                               on='ip-ioa', how='left')
    print(all_categorical.ASDU_Type.unique(), all_categorical.CauseTx.unique())
    # ASDU types: [13, 36, 3, 31, 9, 50, 30, 1, 5]
    asdu_types = pd.get_dummies(all_categorical['ASDU_Type']).rename(
        columns={1: 'asdu1', 3: 'asdu3', 5: 'asdu5', 9: 'asdu9', 13: 'asdu13', 30: 'asdu30', 31: 'asdu31', 36: 'asdu36', 50: 'asdu50'})
    cot_types = pd.get_dummies(all_categorical['CauseTx']).rename(
        columns={1: 'cot1', 2: 'cot2', 3: 'cot3', 6: 'cot6', 7: 'cot7', 20: 'cot20'})
    two_types = pd.concat([asdu_types, cot_types], axis=1)
    samples = pd.concat([samples, two_types], axis=1)
    print('samples: ', samples.shape)
    # add labels
    # input the whitelist
    print('\n******* Prepare dataset *******')
    config_df = pd.read_csv(Path('G:\My Drive', 'IEC104', 'XM_info', 'RTUs_PointsVariables_relabeled.csv'))
    dataset = pd.merge(samples, config_df[['ip-ioa', 'Label']], on='ip-ioa', how='left')
    labels = ['P', 'Q', 'U', 'current', 'frequency', 'AGC-SP (Set-Point)']
    #labels = ['P', 'Q', 'U', 'current', 'frequency']
    dataset = dataset[dataset['Label'].isin(labels)]
    # if manually select partial features
    #dataset = dataset[['ip-ioa', 'Measurement__variance_larger_than_standard_deviation', 'Measurement__median',
    #       'Measurement__mean', 'Label']]
    X = dataset.drop(['ip-ioa', 'Label'], axis=1)
    y = dataset['Label'].astype(str)
    print('X: ', X.shape, 'y: ', y.shape, X.isnull().sum().sum())
    print(y.value_counts())

    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC, SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    models = [
        LogisticRegression(random_state=0),
        DecisionTreeClassifier(random_state=2021),
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(n_estimators=100, max_depth=3, random_state=2021),
        #baseline: XGBClassifier(learning_rate=0.08, max_depth=3, n_estimators=300, subsample=0.8, random_state=2021),
        XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.8, random_state=2021),
        SVC(random_state=2021)
        #LinearSVC(random_state=2021),
        # MultinomialNB(random_state=2021),
    ]

    print('\n******* Prepare train and test sets *******')
    from sklearn.preprocessing import MinMaxScaler
    split_choice = 1

    if split_choice == 1:
        # 1st way to split train and test: random split
        print('Train test split 1: random split')
        X_scaled = MinMaxScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=2021, shuffle=False)
    elif split_choice == 2:
        # 2nd way to split train and test set:
        # Split train and test by rtu-ioa, i.e. test with devices not covered in train
        # devices in each label class pick 80% for training, 20% for testing
        print('Train test split 2: stratified sampling by ip-ioa (device) per label class ')
        label_grouped = dataset.groupby(by=['Label']).groups
        split = 0.8  # portion of samples in train set
        ds_train = pd.DataFrame()
        ds_test = pd.DataFrame()
        for l in label_grouped.keys():
            print('Class ', l)
            cur_l = dataset.loc[label_grouped[l]]   # dataframe in cur label class
            devices = list(cur_l['ip-ioa'].unique())  # all devices in this class
            if len(devices) < 2:
                continue
            #train_devices = devices[:int(len(devices) * split)]
            #test_devices = devices[int(len(devices) * split):]
            train_devices, test_devices = train_test_split(devices, train_size=split, random_state=2022)
            ds_train = pd.concat([ds_train, cur_l[cur_l['ip-ioa'].isin(train_devices)]])
            ds_test = pd.concat([ds_test, cur_l[cur_l['ip-ioa'].isin(test_devices)]])
            #print('New: ', cur_l[cur_l['ip-ioa'] == train_devices[0]].shape)
            #mask = (dataset['Label'] == l) & (dataset['ip-ioa'] == train_devices[0])
            #print('Old: ', dataset[dataset['ip-ioa'] == train_devices[0]].shape)

        X_train = MinMaxScaler().fit_transform(ds_train.drop(['ip-ioa', 'Label'], axis=1))
        y_train = ds_train['Label'].astype(str)
        X_test = MinMaxScaler().fit_transform(ds_test.drop(['ip-ioa', 'Label'], axis=1))
        y_test = ds_test['Label'].astype(str)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print('Training labels: \n', y_train.value_counts())
    print('Testing labels: \n', y_test.value_counts())
    #ds_train.to_csv(Path(out_p, 'ds_train.csv'))


    print('\n******* Start individual model: XGBoost training *******')
    clf = models[4]
    if clf == models[4]:
        clf.fit(X_train, y_train, eval_metric=['auc', 'mlogloss'], eval_set=[(X_train, y_train), (X_test, y_test)]) #, early_stopping_rounds=10)
        evals_result = clf.evals_result()
    else:
        clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_prob_train = clf.predict_proba(X_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    print('Train set mean accuracy: {}, Test set mean accuracy: {}'.format(clf.score(X_train, y_train), clf.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))
    # export prediction result
    #ds_test['Pred'] = y_pred
    #header = ['Class' + str(c) + 'Prob' for c in range(0, y_prob.shape[1])]
    header = [c + 'Prob' for c in clf.classes_]
    # remember to use the same index from ds_test, otherwise, pd.concat() needs to ignore index
    #pred_prob_df = pd.DataFrame(data=y_prob, columns=header, index=ds_test.index)
    #ds_test = pd.concat([ds_test, pred_prob_df], axis=1)
    #ds_test[['ip-ioa', 'Label', 'Pred'] + header].to_csv(Path(out_p, 'bulk_test_w_pred.csv'))
    #ds_test.to_csv(Path(out_p, 'bulk_test_w_pred.csv'))
    #pd.concat([ds_test.iloc[:, 0], ds_test.iloc[:, -8:]], axis=1).to_csv(Path(out_p, 'bulk_test_w_pred.csv'))
    #toContinue = input('Do you want to continue for confusion matrix and feature importance? y/n \n')
    #if toContinue == 'n':
    #    sys.exit()

    # Multi-class confusion matrix
    from sklearn.metrics import multilabel_confusion_matrix
    from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_precision_recall

    # Multi-class confusion matrix
    from sklearn.metrics import multilabel_confusion_matrix
    from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_precision_recall
    cm = multilabel_confusion_matrix(y_test, y_pred)
    #Plotting the confusion matrix
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(y_test, y_pred, x_tick_rotation=45, ax=ax1, figsize=(20, 12))#, normalize=True)
    fig1.savefig(Path(out_p, 'cm.svg'), bbox_inches='tight')
    # feature importance
    sorted_idx = clf.feature_importances_.argsort()
    plt.figure(figsize=(30, 20))
    topk = -20
    plt.barh(X.columns[sorted_idx][topk:], clf.feature_importances_[sorted_idx][topk:])
    plt.xlabel("Feature Importance")
    plt.savefig(Path(out_p, 'fi.svg'), bbox_inches='tight')
    print('the current topk feature importance = ', clf.feature_importances_[sorted_idx][topk:])

    # XGBoost test with top k features
    clf = models[4]
    topk_list = [-20, -30, -50]
    for topk in topk_list:
        if split_choice == 1:
            X_sel = X.loc[:, X.columns[sorted_idx][topk:]]
            X_sel_scaled = MinMaxScaler().fit_transform(X_sel)
            X_sel_train, X_sel_test, y_train, y_test = train_test_split(X_sel_scaled, y, random_state=2021)
        elif split_choice == 2:
            # TODO: implement train test split by points
            X_sel_test = X_test.loc[:, X.columns[sorted_idx][topk:]]
        print(X_sel_train.shape)
        if clf == models[4]:
            clf.fit(X_sel_train, y_train, eval_metric=['auc', 'mlogloss'],
                    eval_set=[(X_sel_train, y_train), (X_sel_test, y_test)],
                    early_stopping_rounds=10)
            evals_result = clf.evals_result()
        else:
            clf.fit(X_sel_train, y_train)
        y_pred = clf.predict(X_sel_test)
        y_prob = clf.predict_proba(X_sel_test)
        print('With top {} features, mean accuracy = {} '.format(np.absolute(topk), clf.score(X_sel_test, y_test)))

    # Cross validation

    print('****** Start cross validation ******')
    CV = 3
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    #i = 0
    #while i < 50:
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    print(entries)
    import seaborn as sns
    fig = plt.figure(figsize=(12, 10))
    #ax = sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    ax = sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.xticks(rotation=30)
    plt.title('accuracy_all')
    plt.savefig(Path(out_p, 'accuracy_allmodels_cv.png'), bbox_inches='tight')

