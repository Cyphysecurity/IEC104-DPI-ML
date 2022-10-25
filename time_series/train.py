'''
## This script:
1. Prepare total dataset with Tsfresh features and labels
2. Choose a specific train and test split method
3. Train the classifiers
4. Present and export results
'''
import logging

from utility import *
import pandas as pd
import numpy as np
import pathlib, time, sys, re, datetime
from pathlib import Path, WindowsPath
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, auc
from sklearn.metrics import multilabel_confusion_matrix
from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_precision_recall


models = [
        LogisticRegression(random_state=0),
        DecisionTreeClassifier(random_state=2021),
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(n_estimators=100, max_depth=3, random_state=2021),
        XGBClassifier(learning_rate=0.08, max_depth=3, n_estimators=50, subsample=0.8, random_state=2021),
        SVC(C=10, random_state=2021),
        SVC(C=20, random_state=2021),
        SVC(C=50, random_state=2021),
        SVC(C=100, random_state=2021)
        #SGDClassifier(random_state=2021),
    ]

def ds_prep(df, config_fpath, logger: logging.getLogger()):
    '''
    Prepare total dataset with features and labels
    Features:
        1. TSFRESH
        2. 104 features
    '''
    # add labels
    # input the whitelist
    #print('\n******* Prepare total dataset *******', file=log_outf)
    logger.info('\n******* Prepare total dataset *******')
    samples = pd.read_csv(config_fpath).rename(columns={'ip-ioa': 'rtu-ioa'}) # tsfresh features with RTU IDs
    config_path = Path('G:\My Drive', 'IEC104', 'Netherlands-results', 'gas_dataset_point_configuration_labeled.csv')
    config_df = pd.read_csv(config_path)
    dataset = pd.merge(samples, config_df[['rtu-ioa', 'Label']], on='rtu-ioa', how='left')
    labels = ['alarm_config', 'flow', 'pressure', 'position', 'temperature'] # drop those labels with very few samples
    dataset = dataset[dataset['Label'].isin(labels)]    # tsfresh features with RTU IDs and labels

    # IEC 104 categorical features processing
    all_categorical = pd.merge(dataset['rtu-ioa'],
                               df[['rtu-ioa', 'ASDU_Type', 'CauseTx']].drop_duplicates(subset=['rtu-ioa']),
                               on='rtu-ioa', how='left')
    #print(all_categorical.ASDU_Type.unique(), all_categorical.CauseTx.unique(), file=log_outf)
    logger.info('ASDU types: {}, CoT: {}'.format(all_categorical.ASDU_Type.unique(), all_categorical.CauseTx.unique()))
    # ASDU types: [1, 9, 30, 34, 48]
    asdu_types = pd.get_dummies(all_categorical['ASDU_Type']).rename(
        columns={1: 'asdu1', 9: 'asdu9', 30: 'asdu30', 34: 'asdu34', 48: 'asdu48'})
    cot_types = pd.get_dummies(all_categorical['CauseTx']).rename(
        columns={1: 'cot1', 2: 'cot2', 3: 'cot3', 5: 'cot5', 6: 'cot6', 7: 'cot7', 12: 'cot12', 20: 'cot20'})
    two_types = pd.concat([asdu_types, cot_types], axis=1).set_index(dataset.index)
    dataset = pd.concat([dataset, two_types], axis=1)
    X = dataset.drop(['rtu-ioa', 'Label'], axis=1)
    y = dataset['Label'].astype(str)
    #print('X: ', X.shape, 'y: ', y.shape, X.isnull().sum().sum(), file=log_outf)
    #print(y.value_counts(), file=log_outf)
    logger.info('X: {}, y: {}, null # in X: {}'.format(X.shape, y.shape, X.isnull().sum().sum()))
    #print('\n******* Downscale dataset to mitigate dominant classes *******', file=log_outf)
    logger.info('\n******* Downscale dataset to mitigate dominant classes *******')
    # downsample dominating classes
    # ya_list = [ya_config, ya_flow, ya_position, ya_pres, ya_temp]
    labels = ['alarm_config', 'pressure']  # ['alarm_config', 'flow', 'position', 'pressure', 'temperature']
    downFrac = 0.1
    #print('Downscale rate: ', downFrac, file=log_outf)
    logger.info('Downscale rate = {}'.format(downFrac))
    for l in labels:
        y_tmp = y[y == l]
        #print('Label {} before downsample: {} '.format(l, y_tmp.shape), file=log_outf)
        logger.info('Label {} before downsample: {} '.format(l, y_tmp.shape))
        y_tmp = y_tmp.sample(frac=downFrac, random_state=2021)
        #print('After downsample: ', y_tmp.shape, file=log_outf)
        logger.info('After downsample: {}'.format(y_tmp.shape))
        y = pd.concat([y[y != l], y_tmp])
        #print('After downsample {}, y: {}'.format(l, y.shape), file=log_outf)
        logger.info('After downsample {}, y: {}'.format(l, y.shape))
    X = X.loc[y.index, :]
    dataset = dataset.loc[y.index, :]
    #print('X shape: ', X.shape, file=log_outf)
    logger.info('X shape: {}'.format(X.shape))
    return dataset, X, y


def train_test_prep(dataset, X, y, logger: logging.getLogger()):
    '''
    2. Choose a specific train and test split method
    '''
    #print('\n******* Prepare train and test sets *******', file=log_outf)
    logger.info('\n******* Prepare train and test sets *******')
    ds_train = pd.DataFrame()
    ds_test = pd.DataFrame()
    split_choice = 1
    if split_choice == 1:
        # 1st way to split: split first k folds as training, last k folds as testing, stratified by classes
        #print('Train test split 1: stratified split in time sequence', file=log_outf)
        logger.info('Train test split 1: stratified split in time sequence')
        X = X.drop(['Unnamed: 0'], axis=1)
        X_scaled = MinMaxScaler().fit_transform(X)

        skf = StratifiedKFold(n_splits=4)
        idx = 0
        for train_index, test_index in skf.split(X_scaled, y):
            #print(idx, " iteration\n", file=log_outf)
            logger.info('{} iteration\n'.format(idx))
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            idx += 1

        #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=2021, shuffle=True)
        ds_train = pd.DataFrame(data=X_train, columns=X.columns, index=y_train.index)
        # insert RTU IDs and labels from dataset
        ds_train.insert(0, 'rtu-ioa', dataset.loc[ds_train.index]['rtu-ioa'])
        ds_train.insert(ds_train.shape[1], 'Label', dataset.loc[ds_train.index]['Label'])
        ds_test = pd.DataFrame(data=X_test, columns=X.columns, index=y_test.index)
        ds_test.insert(0, 'rtu-ioa', dataset.loc[ds_test.index]['rtu-ioa'])
        ds_test.insert(ds_test.shape[1], 'Label', dataset.loc[ds_test.index]['Label'])


    elif split_choice == 2:
        # 2nd way to split train and test set:
        # Split train and test by rtu-ioa, i.e. test with devices not covered in train
        # devices in each label class pick 80% for training, 20% for testing
        #print('Train test split 2: stratified sampling by ip-ioa (device) per label class ', file=log_outf)
        logger.info('Train test split 2: stratified sampling by ip-ioa (device) per label class ')
        label_grouped = dataset.groupby(by=['Label']).groups
        split = 0.8  # portion of samples in train set

        for l in label_grouped.keys():
            cur_l = dataset.loc[label_grouped[l]]  # dataframe in cur label class
            devices = list(cur_l['rtu-ioa'].unique())  # all devices in this class
            if len(devices) < 2:
                continue
            # train_devices = devices[:int(len(devices) * split)]
            # test_devices = devices[int(len(devices) * split):]
            train_devices, test_devices = train_test_split(devices, train_size=split, random_state=2022)
            ds_train = pd.concat([ds_train, cur_l[cur_l['rtu-ioa'].isin(train_devices)]])
            ds_test = pd.concat([ds_test, cur_l[cur_l['rtu-ioa'].isin(test_devices)]])
            # print('New: ', cur_l[cur_l['ip-ioa'] == train_devices[0]].shape)
            # mask = (dataset['Label'] == l) & (dataset['ip-ioa'] == train_devices[0])
            # print('Old: ', dataset[dataset['ip-ioa'] == train_devices[0]].shape)

        X_train = MinMaxScaler().fit_transform(ds_train.drop(['rtu-ioa', 'Label'], axis=1))
        y_train = ds_train['Label'].astype(str)
        X_test = MinMaxScaler().fit_transform(ds_test.drop(['rtu-ioa', 'Label'], axis=1))
        y_test = ds_test['Label'].astype(str)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, file=log_outf)
    #print('Training labels: \n', y_train.value_counts(), file=log_outf)
    #print('Testing labels: \n', y_test.value_counts(), file=log_outf)
    logger.info('Shapes of X_train = {}, X_test = {}, y_train = {}, y_test = {}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    logger.info('Training labels: {}\n'.format(y_train.value_counts()))
    logger.info('Testing labels: {}\n'.format(y_test.value_counts()))
    return ds_train, X_train, y_train, ds_test, X_test, y_test


def model_train(ds_train, X_train, y_train, ds_test, X_test, y_test, out_fp, logger:logging.getLogger()):
    '''
    3. Train the classifiers
    '''
    #print('\n******* Start individual model training *******', file=log_outf)
    logger.info('\n******* Start individual model training *******')
    clf = models[0]
    logger.info('Model: {}'.format(clf))
    if clf == models[4]:
        #clf.fit(X_train, y_train, eval_metric=['auc', 'mlogloss'], eval_set=[(X_train, y_train), (X_test, y_test)],
         #       early_stopping_rounds=10)
        clf.fit(X_train, y_train, eval_metric=['auc', 'mlogloss'], eval_set=[(X_train, y_train), (X_test, y_test)])
        #clf.fit(X_train, y_train)

    else:
        clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_prob_train = clf.predict_proba(X_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    #print('Train set mean accuracy: {}, Test set mean accuracy: {}'.format(clf.score(X_train, y_train), clf.score(X_test, y_test)), file=log_outf)
    logger.info('Train set mean accuracy: {}, Test set mean accuracy: {}'.format(clf.score(X_train, y_train),
                                                                           clf.score(X_test, y_test)))
    # export prediction result
    ds_test['Pred'] = y_pred
    header = [c + 'Prob' for c in clf.classes_]
    # remember to use the same index from ds_test, otherwise, pd.concat() needs to ignore index
    pred_prob_df = pd.DataFrame(data=y_prob, columns=header, index=ds_test.index)
    ds_test = pd.concat([ds_test, pred_prob_df], axis=1)
    ds_test[['rtu-ioa', 'Label', 'Pred'] + header].to_csv(out_fp)
    #pd.concat([ds_test.iloc[:, 0], ds_test.iloc[:, -8:]], axis=1).to_csv(Path(out_p, 'bulk_test_w_pred.csv'))
    #toContinue = input('Do you want to continue for confusion matrix and feature importance? y/n \n')
    #if toContinue == 'n':
    #    sys.exit()
    return clf, y_pred, y_prob


def get_results(clf, X, y, y_test, y_pred, y_prob, out_p, logger: logging.getLogger()):
    '''
    4. Present and export classification results
    '''
    logger.info('***** Classification Results *****')
    # Multi-class confusion matrix
    from sklearn.metrics import multilabel_confusion_matrix
    from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_precision_recall

    # Multi-class confusion matrix
    cm = multilabel_confusion_matrix(y_test, y_pred)
    # Plotting the confusion matrix
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(y_test, y_pred, x_tick_rotation=45, ax=ax1, figsize=(20, 12))  # , normalize=True)
    #print(classification_report(y_test, y_pred), file=log_outf)
    logger.info(classification_report(y_test, y_pred))
    fig1.savefig(Path(out_p, 'cm.svg'), bbox_inches='tight')
    logger.info('Export confusion matrix to {}'.format(Path(out_p, 'cm.svg')))
    # ROC and precision-recall curves
    fig2, ax2 = plt.subplots()
    plot_roc(y_test, y_prob, ax=ax2, figsize=(20, 12))
    fig2.savefig(Path(out_p, 'roc.svg'), bbox_inches='tight')
    logger.info('Export roc curve to {}'.format(Path(out_p, 'roc.svg')))
    fig3, ax3 = plt.subplots()
    plot_precision_recall(y_test, y_prob, ax=ax3, figsize=(20, 12))
    fig3.savefig(Path(out_p, 'pr.svg'), bbox_inches='tight')
    logger.info('Export precision-recall curve to {}'.format(Path(out_p, 'pr.svg')))


    # feature importance
    sorted_idx = clf.feature_importances_.argsort()
    plt.figure(figsize=(30, 20))
    topk = -20
    plt.barh(X.columns[sorted_idx][topk:], clf.feature_importances_[sorted_idx][topk:], color='#019267')
    plt.xlabel("Feature Importance")
    plt.savefig(Path(out_p, 'fi_top{}.svg'.format(topk)), bbox_inches='tight')
    #print('the current topk feature importance = ', clf.feature_importances_[sorted_idx][topk:], file=log_outf)
    logger.info('the current topk feature importance = {}'.format(clf.feature_importances_[sorted_idx][topk:]))
    # export top features
    with open(Path(out_p, 'fi_info.txt'), 'a') as fi_outf:
        fi_outf.write('Top {} feature importance: \n'.format(topk))
        for col, score in zip(X.columns[sorted_idx][topk:], clf.feature_importances_[sorted_idx][topk:]):
            fi_outf.write(col + '\t' + str(score) + '\n')
        fi_outf.close()


def prune_by_features(X, y, sorted_idx, out_p, logger):
    """
    # 5. (optional) Prune features through XGBoost performance test with different top k features
    """
    logger.info('****** Prune features ******')
    clf_top_test = models[4]
    topk_list = [-20, -30, -50]
    for topk in topk_list:
        X_sel = X.loc[:, X.columns[sorted_idx][topk:]]
        X_sel_scaled = MinMaxScaler().fit_transform(X_sel)
        X_sel_train, X_sel_test, y_train, y_test = train_test_split(X_sel_scaled, y, random_state=2021)
        #print(' X_sel_train shape: {}'.format(X_sel_train.shape), file=log_outf)
        logger.info('X_sel_train shape: {}'.format(X_sel_train.shape))
        clf_top_test.fit(X_sel_train, y_train)
        cur_y_pred = clf_top_test.predict(X_sel_test)
        cur_y_prob = clf_top_test.predict_proba(X_sel_test)
        message = 'With top {} features, classification report:\n{}\n'.format(np.absolute(topk), classification_report(y_test, cur_y_pred))
        #print(message, file=log_outf)
        logger.info(message)
        with open(Path(out_p, 'fi_info.txt'), 'a') as fi_outf:
            fi_outf.write(message)
            fi_outf.close()


def multi_train(ds_train, X_train, y_train, ds_test, X_test, y_test, out_p, logger: logging.getLogger()):
    # Run multiple algorithms
    #print('****** Start trying all classifier algorithms ******', file=log_outf)
    logger.info('****** Start trying all classifier algorithms ******')
    '''
        >> > kf = KFold(n_splits=2)
        >> > kf.get_n_splits(X)
        2
        >> > print(kf)
        KFold(n_splits=2, random_state=None, shuffle=False)
        >> > for train_index, test_index in kf.split(X):
            ...
            print("TRAIN:", train_index, "TEST:", test_index)
        ...
        X_train, X_test = X[train_index], X[test_index]
        ...
        y_train, y_test = y[train_index], y[test_index]
    
    '''

    for model in models:
        model_name = model.__class__.__name__
        #print('Model: ', model_name, file=log_outf)
        logger.info('Model: {}'.format(model_name))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        message = 'Train set mean accuracy: {}, Test set mean accuracy: {}\n'.format(model.score(X_train, y_train),
                                                                               model.score(X_test, y_test))
        message += classification_report(y_test, y_pred) + '\n'
        if model != models[5]:
            y_prob = model.predict_proba(X_test)
            message += 'ROC AUC: ' + str(roc_auc_score(y_test, y_prob, multi_class='ovr')) + '\n'

        #print(message, file=log_outf)
        logger.info(message)
        # Plotting the confusion matrix
        fig1, ax1 = plt.subplots()
        plot_confusion_matrix(y_test, y_pred, x_tick_rotation=45, ax=ax1, figsize=(20, 12))  # , normalize=True)
        fig1_p = Path(out_p, str(model)[:10] + '_cm.pdf')
        fig1.savefig(fig1_p, bbox_inches='tight')
        logger.info('Export confusion matrix to {}'.format(fig1_p))
        # ROC and precision-recall curves
        fig2, ax2 = plt.subplots()
        plot_roc(y_test, y_prob, ax=ax2, figsize=(20, 12))
        fig2_p = Path(out_p, str(model)[:10] + '_roc.pdf')
        fig2.savefig(fig2_p, bbox_inches='tight')
        logger.info('Export roc curve to {}'.format(fig2_p))
        fig, ax = plt.subplots()
        plot_precision_recall(y_test, y_prob, ax=ax, figsize=(20, 12))
        fig_p = Path(out_p, str(model)[:10] + '_pr.pdf')
        fig.savefig(fig_p, bbox_inches='tight')
        logger.info('Export precision-recall curve to {}'.format(fig_p))
    # Uncomment to get Cross validation
    '''
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
    '''
    return
