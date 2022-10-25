import pandas as pd
import numpy as np
import pathlib, time, sys, re
from pathlib import Path, WindowsPath
import matplotlib.pyplot as plt

# TODO: clean to run
'''
Label point variable type
Test group 1 labels:
Analog:
alarm configuration, gas flow rate control, pressure, temperature, flow rate
Digital:
alarm status, regulator control, expansion emergency control, valve, position, membrane, motor, test
Step 1: get labels per RTU-IOA from config file
'''


def regex_desc(pattern, string):
    return re.search(pattern, string, re.IGNORECASE)


def parse_desc(row_pointId, row_desc):
    label = ''
    if row_pointId[0] == 'D':
        # alarm status, regulator control, expansion emergency control,
        # valve, position, membrane, motor, test
        if regex_desc('(alarm|fault)', row_desc):
            label = 'alarm'
        elif regex_desc('emergency', row_desc):
            label = 'em_ctrl'
        elif regex_desc('test', row_desc):
            label = 'test'
        else:
            if regex_desc('regulator', row_desc):
                label = 'regulator'
            elif regex_desc('valve', row_desc):
                label = 'valve'
            elif regex_desc('pressure', row_desc):
                label = 'pressure'
            elif regex_desc('membrane', row_desc):
                label = 'membrane'
            elif regex_desc('motor', row_desc):
                label = 'motor'
            # updated
            elif regex_desc('last operation', row_desc):
                label = 'last_op'
            elif regex_desc('MCU', row_desc) or regex_desc('ECU', row_desc):
                label = 'microcontroller'
            elif regex_desc('El-O-Matic', row_desc):
                label = 'ei-o-matic_actuator'
            elif regex_desc('Reboot', row_desc):
                label = 'reboot'
            elif regex_desc('Reset', row_desc):
                label = 'reset_status'
            else:
                label = 'Undecided'
    elif row_pointId[0] == 'A':
        # alarm configuration, gas flow rate control,  flow rate, pressure, temperature
        if regex_desc('limit', row_desc):
            label = 'alarm_config'
        elif regex_desc('setpoint', row_desc):
            label = 'gas_flow_config'
        else:
            if regex_desc('flow', row_desc):
                label = 'flow'
            elif regex_desc('pressure', row_desc):
                label = 'pressure'
            elif regex_desc('position', row_desc):
                label = 'position'
            elif regex_desc('temperature', row_desc):
                label = 'temperature'
            elif regex_desc('client', row_desc):
                label = 'clients'
            # updated
            elif regex_desc('Repl.', row_desc):
                label = 'repl.'
            elif regex_desc('Mag Field', row_desc):
                label = 'magnetic_field'
            elif regex_desc('Post Regulation', row_desc):
                label = 'post_regulation'
            else:
                label = 'Undecided'

    return label


def point_labeling_rule():
    '''
    set up the rules for labeling
    '''
    config_path = Path('G:\My Drive', 'IEC104', 'Netherlands-results', 'gas_dataset_point_configuration_recolumned.csv')
    config_df = pd.read_csv(config_path)
    config_df['rtu-ioa'] = config_df['RTU'].astype(str) + '-' + config_df['Monitor IOA'].astype(str)

    config_df['Label'] = config_df.apply(lambda x: parse_desc(x['Point ID'], x['Description']), axis=1)
    undecided_df = config_df[config_df['Label'] == 'Undecided']
    print('There are {} rows remaining to be label-undecided'.format(undecided_df.shape[0]))
    print('*** Distribution of variable types ***')
    print(config_df['Label'].value_counts())
    return config_df


def print_var_dist(config_df):
    print(config_df['Label'].value_counts())


def export_config(config_df, p):
    # config_df.to_csv(Path('G:\My Drive', 'IEC104', 'Netherlands-results', 'gas_dataset_point_configuration_labeled.csv'), index=False)
    config_df.to_csv(p, index=False)


def import_config(p):
    return pd.read_csv(p)
#config_df = pd.read_csv(Path('G:\My Drive', 'IEC104', 'Netherlands-results', 'gas_dataset_point_configuration_labeled.csv'))

