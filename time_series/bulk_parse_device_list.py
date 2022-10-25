'''
This script parses the provided whitelist of devices and give new labels
'''
import re
from pathlib import Path

import pandas as pd


def regex_desc(pattern, string):
    return re.search(pattern, string, re.IGNORECASE)

#
def parse_desc(type_name):
    label = ''
    if type_name.startswith('I'):   # I, I-A/B/C
        label = 'current'
    elif regex_desc('Gen', type_name):
        label = 'gen'
    elif type_name == 'Frequ':
        label = 'frequency'
    else:
        label = type_name
    return label


# set up the rules for labeling
def point_labeling_rule(config_path):
    #config_path = Path('G:\My Drive', 'IEC104', 'Netherlands-results', 'gas_dataset_point_configuration_recolumned.csv')
    if str(config_path).split('.')[1] == 'csv':
        config_df = pd.read_csv(config_path)
    elif str(config_path).split('.')[1] == 'xlsx':
        config_df = pd.read_excel(config_path)
    config_df['ip-ioa'] = config_df['IP'].astype(str) + '-' + config_df['IOA'].astype(str)
    config_df['Type Variable'] = config_df['Type Variable'].str.strip()
    print(config_df['Type Variable'].unique())
    config_df['Label'] = config_df.apply(lambda x: parse_desc(x['Type Variable']), axis=1)
    undecided_df = config_df[config_df['Label'] == "-"]
    print('There are {} rows remaining to be label-undecided. Temporarily drop \'undecided\' labels if any'.format(
        undecided_df.shape[0]))
    print('*** Distribution of variable types ***')
    print(config_df['Label'].value_counts())
    return config_df[config_df['Label'] != 'Undecided']



# config_df.to_csv(Path('G:\My Drive', 'IEC104', 'Netherlands-results', 'gas_dataset_point_configuration_labeled.csv'), index=False)
if __name__ == '__main__':
    # I/O
    cur_p = Path().absolute()
    # G:\My Drive\IEC104\XM-results\IOA_timeseries\bulk2017
    in_p = Path('G:\My Drive', 'IEC104', 'XM-results', 'IOA_timeseries', 'bulk2017')
    out_p = Path(Path().absolute().parents[1], 'output')
    print('Current path: {}\nInput path: {}\nOutput path: {}'.format(cur_p, in_p, out_p))
    # XM_info: RTUs_PointsVariables(1).xlsx
    config_path = Path('G:\My Drive', 'IEC104', 'XM_info', 'RTUs_PointsVariables(1).xlsx')
    config_df = point_labeling_rule(config_path)
    print(config_df.shape, config_df.head(2), config_df.columns)
    for empty_col in config_df.columns:
        if empty_col.startswith('Unnamed'):
            print(empty_col)
            config_df.drop([empty_col], axis=1, inplace=True)
            print('after drop: ', config_df.shape)
    config_df.to_csv(Path('G:\My Drive', 'IEC104', 'XM_info', 'RTUs_PointsVariables_relabeled.csv'), index=False)
