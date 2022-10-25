'''
## This script runs for GAS:
1. categorical feature engineering
2. add labels

## Supervised learning pipeline
Step 1: var_config_parse.py -> generate tidied point labels
Step 2: tsfresh_features.py -> generate TSFRESH features, clean null & infinite, and export for storage
Step 3: train.py -> supervised model train and test
'''
from utility import *
from train import *
from tsfresh_features import *
from breakpoints import *
from post_classify import extract_misclassify, scatter_one, manifold_umap, plot_shap_summary, plot_shap_force
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
from sklearn.pipeline import Pipeline
import logging


def io():
    '''
    input time series data from IEC 104 parser
    Get all the time series data per RTU and merge (from Google drive folder)
    '''
    cur_p = Path().absolute()
    out_p = Path(Path().absolute().parents[1], 'output')
    print('Output path: {}\n'.format(out_p))
    return cur_p, out_p


def input_collect(in_p: pathlib.Path, logger: logging.getLogger()) -> pd.DataFrame:
    df = pd.DataFrame()
    cnt = 0
    if not in_p.exists():
        logger.ERROR('Google Drive may not be active!!!')
        sys.exit()
    for f in in_p.glob('*'):
        # print(f)
        cur_df = pd.read_csv(f)
        cnt += cur_df.shape[0]
        df = pd.concat([df, cur_df])
    df['rtu-ioa'] = df['ASDU_addr'].astype(str) + '-' + df['IOA'].astype(str)
    logger.info('After concatenation, data row # equal to summing up all the individual dataframes? {} {}'.format(cnt == df.shape[0],
          df.shape[0]))
    return df


def divide_ad(df, a_types, d_types):
    '''
    divide data into analog/digital variables based on ASDU types
    # ASDU types: [9, 1, 34, 48, 45, 30]
    '''
    print('complet: ', df.shape)
    df_analog = df[df['ASDU_Type'].isin(a_types)]
    print('Choose analog: ', df_analog.shape)
    df_digital = df[df['ASDU_Type'].isin(d_types)]
    print('Choose digital', df_digital.shape)
    return df_analog, df_digital


def add_day_cols(df):
    """
    Convert epoch timestamp to days and human readable time columns
    """
    df['datetime'] = df.apply(lambda x: datetime.datetime.fromtimestamp(x['Time']), axis=1)
    df['Day'] = df.apply(lambda x: x['datetime'].day, axis=1)
    print(df.head())
    return df


def logging_setup(log_fp: pathlib.Path) -> logging.getLogger():
    """
    The configurations for customized logger
    Separate handlers for console print-out and log file
    """
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler_format = '%(levelname)s: [%(filename)s] - %(message)s'
    console_handler.setFormatter(logging.Formatter(console_handler_format))
    logger.addHandler(console_handler)
    # log file handler
    file_handler = logging.FileHandler(log_fp)
    file_handler.setLevel(logging.DEBUG)
    file_handler_format = '[%(levelname)s] - [%(filename)s > %(funcName)s() > %(lineno)s] - %(message)s'
    file_handler.setFormatter(logging.Formatter(file_handler_format))
    logger.addHandler(file_handler)
    return logger
    #logging.basicConfig(level=logging.DEBUG,
     #               logger=mylogs,
      #              fmt='%(asctime)s [%(levelname)s] - [%(filename)s > %(funcName)s() > %(lineno)s] - %(message)s',
       #             datefmt='%H:%M:%S')


def main():
    cur_p, out_p = io()
    in_p = Path('G:\My Drive', 'IEC104', 'Netherlands-results', 'Ioa_timeseries',
                'rtus-22-26-172-228-230-11024-11027_alltime')  # 'known_rtus-22-26-92-172_alltime') 'known_rtus_rest_alltime')  # '04-15')
    # create a timestamped output folder
    cur_time = str(time.strftime("%b-%d-%Y-%H-%M-%S", time.localtime()))
    out_subpath = 'ts_main_' + cur_time
    out_subpath = Path(out_p, out_subpath)
    out_subpath.mkdir(parents=True, exist_ok=True)
    # create a log file to record all the prints to screen under this function
    #log_outf = open(Path(out_subpath, 'log.txt'), 'a')
    log_fp = Path(out_subpath, 'log.txt')
    mylog = logging_setup(log_fp)
    mylog.info('Input path: {}'.format(in_p))
    df = input_collect(in_p, mylog)
    print_info(df)

    #print('new sub-folder for output files is created: ', out_subpath)
    mylog.info('new sub-folder for output files is created: {}'.format(out_subpath))
    choice = int(input('Choose the step to execute: \n' \
                       '1. Extract Tsfresh time series features\n' \
                       '2. Parse point variable label config file (To implement)\n' \
                       '3. Train multiple classifiers\n' \
                       '4. Train XGBoost classifier (include SHAP)\n' \
                       '5. Find breakpoints\n' \
                       '6. Post-classification analysis\n' \
                       '7. Denormalize value\n'))
    mylog.info('***** Log for time series main function {} run *****\n'.format(choice))
    if choice in [1, 2, 3, 4, 5, 6]:
        ts_extracted_f = Path(out_p, 'gas_samples_extracted_n=60.csv')  # TODO: Better IO naming and folder management
        mylog.info('The file of extracted TSFRESH features is: ' + str(ts_extracted_f))
        dataset, X, y = ds_prep(df, ts_extracted_f, mylog)

    # TODO: choice 2 to move from notebook
    # TODO: add output log (log_outf) and timestamped output folder (out_subpath) to function 1, 2, 5, 7
    if choice == 1:
        in_p = Path('G:\My Drive', 'IEC104', 'Netherlands-results', 'Ioa_timeseries',
                    'rtus-22-26-172-228-230-11024-11027_alltime')  # 'known_rtus-22-26-92-172_alltime') 'known_rtus_rest_alltime')  # '04-15')
        print('Input path: ', in_p)
        df = input_collect(in_p)
        print_info(df)
        a_types = [9, 48]
        d_types = [1, 34, 30]
        df_analog, df_digital = divide_ad(df, a_types, d_types)
        df_analog = add_day_cols(df_analog)
        # precompute the tsfresh features for all data
        extract_tsfresh(df_analog, out_p)
    elif choice == 3:
        ds_train, X_train, y_train, ds_test, X_test, y_test = train_test_prep(dataset, X, y, mylog)
        multi_train(ds_train, X_train, y_train, ds_test, X_test, y_test, out_subpath, mylog)
    elif choice == 4:
        # TODO: whether to add model preserve function
        ds_train, X_train, y_train, ds_test, X_test, y_test = train_test_prep(dataset, X, y, mylog)
        pred_fp = Path(out_subpath, 'gas_test_w_pred.csv')
        mylog.info('prediction result file is: ' + str(pred_fp))
        trained_model, y_pred, y_prob = model_train(ds_train, X_train, y_train, ds_test, X_test, y_test, pred_fp, mylog)
         # SHAP plot
        #X = X.rename(columns={'Measurement_absolute_sum_of_changes': 'Sum of absolute changes'})
        # rename: {'Measurement_absolute_sum_of_changes':'Sum of absolute changes', 'Measurement_change_quantiles_f_agg_\"mean\"_isabs_True_qh_0.6_ql_0.0': 'Mean in the quantiles of [0.0, 0.6]', 'Measurement_lempel_ziv_complexity_bins_100': 'Lempel-Ziv complexity (bins=100)', 'Measurement_sum_of_reoccurring_values': 'Sum of reoccuring values', 'Measurement_maximum': 'Maximum value', 'Measurement_minimum': 'Minimum value', 'Mean in the quantiles of [0.0, 0.8]', 'Measurement_quantile_q_0.1': 'Value at quantile 0.1', 'Measurement_quantile_q_0.4': 'Value at quantile 0.4', 'asdu9': 'ASDU normalized measurement type'}
        features = X.drop(['Unnamed: 0'], axis=1).columns
        #sorted_idx = trained_model.feature_importances_.argsort()
        #prune_by_features(X, y, sorted_idx, out_subpath, mylog)
        in_f = 'gas_test_w_pred.csv'
        #totaldf, misdf = extract_misclassify(Path(out_p, in_f))
        #mis_id = '22-9220' # line 757, 766
        label_class = 'position'
        pred_class = 'alarm_config'
        # filter data for mis_id, all correct label class and all correct pred class
        #single_mis_df = misdf[misdf['rtu-ioa'] == mis_id]
        #single_mis_x = X.loc[single_mis_df.index]
        #print(single_mis_x.shape)
        #plot_shap_force(trained_model, X_test, features)
        plot_shap_summary(trained_model, X, X_train, X_test, features, out_subpath, mylog)
        get_results(trained_model, X, y, y_test, y_pred, y_prob, out_subpath, mylog)
    elif choice == 5:
        # G:\My Drive\IEC104\Netherlands-results\IOA_timeseries\Emmanuele_start
        in_p = Path('G:\My Drive', 'IEC104', 'Netherlands-results', 'IOA_timeseries', 'Emmanuele_start', 'rtu_172_point_data.csv')
        df = bp_input(in_p) # total data from input file
        rtu = 172
        ioa = 8193
        val_max = 160
        df['denorm_val'] = np.divide(np.multiply(val_max, np.add(1, df['VAL'])), 2)
        # header: TIMESTAMP, CA, IOA, VAL
        point_id = str(rtu) + "-" + str(ioa)
        dft = df[df['IOA'] == ioa]  # filter certain IOA time series
        dft[['TIMESTAMP', 'denorm_val']].hist()
        sys.exit()
        print('Selected data shape: ', dft.shape)
        logging.info('Selected data shape: '.format(dft.shape))

        bp_params = {
            'n_break': 3,
            'submodel': 'l2'
        }
        is_bp_idx_generated = True
        if is_bp_idx_generated:
            bp_idx_f = Path(out_p, 'breakpoints-bp-idx_point=172-8193_model=dynp_submodel=l2_bpNum=5.csv')
            print('Breakpoints idx import from: ', bp_idx_f)
            bp_idx = pd.read_csv(bp_idx_f)['0'].to_list()
        else:
            bp_idx, bp_model = apply_model(dft['denorm_val'].values, bp_params['n_break'], bp_params['submodel'])
            out_f = 'breakpoints-bp-idx_point={}_model={}_submodel={}_bpNum={}.csv'.format(point_id, \
                str(bp_model).strip("<").strip(">").split(".")[2], bp_params['submodel'], bp_params['n_break'])
            print('Breakpoints idx exported to: ', out_f)
            pd.DataFrame(data=bp_idx).to_csv(Path(out_p, out_f))
        bp_idx = np.subtract(bp_idx, 1)
        bp_values = plot_seg(dft['denorm_val'].values, bp_idx)
        intervals = get_ci(dft['denorm_val'].values, bp_idx, 0.05)
        print('Breakpoint values: ', bp_values)
        print('Intervals are: ', intervals)
        anomaly_idx = get_anomaly(dft['denorm_val'].values, bp_idx, intervals)
        print('There are {} anomalies.'.format(len(anomaly_idx)))
        plot_anomaly(dft['denorm_val'], anomaly_idx)
        #anomaly_df = pd.DataFrame(columns=['idx', 'value'])
        #for idx in anomaly_idx:
            #print(idx, dft['denorm_val'].iloc[idx])
            #anomaly_df.append({'idx': idx, 'value': dft['denorm_val'].iloc[idx]})
        #print(anomaly_df)

        #prophet_params = {
        #    'interval_width': 0.95,
        #    'changepoint_range': 0.8
        #}
        #prophet_ad(dft[['TIMESTAMP', 'denorm_val']], params=prophet_params)
    elif choice == 6:
        in_f = 'gas_test_w_pred.csv'
        X = X.drop(['Unnamed: 0'], axis=1)
        df, misdf, mis_count_df = extract_misclassify(Path(out_p, in_f), mylog)
        mis_count_df.to_csv(Path(out_subpath, 'misclassify_per_var.csv'))
        # top wrong: label position v.s. pred alarm config, 22-9220
        # second wrong: label temperature v.s. pred pressure, 228-8194
        mis_id = '22-9220'
        label_class = 'position'
        pred_class = 'alarm_config'
        # filter data for mis_id, all correct label class and all correct pred class
        single_mis_df = misdf[misdf['rtu-ioa'] == mis_id]
        single_mis_x = X.loc[single_mis_df.index]
        # correctly classified samples
        label_mask = (df['Label'] == df['Pred']) & (df['Label'] == label_class)
        pred_mask = (df['Label'] == df['Pred']) & (df['Label'] == pred_class)
        label_df = df[label_mask]
        pred_df = df[pred_mask]
        #print('Point {} has {} samples mis-classified, correct: label {} has {}, label {} has {}'.format(
         #   mis_id, single_mis_df.shape[0], label_class, label_df.shape[0], pred_class, pred_df.shape[0]), file=log_outf)
        mylog.info('Point {} has {} samples mis-classified, correct: label {} has {}, label {} has {}'.format(
            mis_id, single_mis_df.shape[0], label_class, label_df.shape[0], pred_class, pred_df.shape[0]))
        label_x = X.loc[label_df.index]
        pred_x = X.loc[pred_df.index]
        #manifold_umap(mis_id, single_mis_x, label_class, label_x, pred_class, pred_x, out_subpath, mylog)
    elif choice == 7:
        rtu172 = df[df['ASDU_addr'] == 172]
        ioa8193 = rtu172[rtu172['IOA'] == 8193]
        print('IOA 8193 has {} rows '.format(ioa8193.shape[0]))
        ioa8193['Denormed'] = denormalize_val_col(ioa8193['Measurement'], -40, 160)
        ioa8213 = rtu172[rtu172['IOA'] == 8213]
        print('IOA 8213 has {} rows'.format((ioa8213.shape[0])))
        ioa8213['Denormed'] = denormalize_val_col(ioa8213['Measurement'], -25, 100)
        ioa88 = rtu172[rtu172['IOA'] == 88]
        print('IOA 88 has {} rows'.format(ioa88.shape[0]))
        out_f = 'rtu172_ioa8193_ioa8213_ioa88_timeseries.csv'
        out_df = pd.concat([ioa8193, ioa8213, ioa88])
        print('out_df: ', out_df.shape)
        out_df.to_csv(Path(out_p, out_f))
    else:
        mylog.error('Wrong input for choice number!!!')
    #print('****** Main function finished!!! *******', file=log_outf)
    mylog.info('****** Main function finished!!! *******')
    #log_outf.close()

if __name__ == '__main__':
    main()
