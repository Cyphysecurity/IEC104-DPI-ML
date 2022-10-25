import pandas as pd
import numpy as np
import pathlib, time, sys, re
from pathlib import Path, WindowsPath
import matplotlib.pyplot as plt


def bp_input(in_p: Path) -> pd.DataFrame:
    """
    Import time series CSV file for breakpoint analysis
    Parameter:
        in_p: Path in pathlib
    Returns:
        a Pandas data frame
    """
    df = pd.read_csv(in_p)
    # header: TIMESTAMP, CA, IOA, VAL
    print('Import from: {}\nShape: {}'.format(in_p, df.shape))
    return df


def print_info(df):
    print('In this input data: \n')
    print('df shape: ', df.shape)
    print('df column header: ', df.columns)
    print('RTUs/SCADA server: {} \nASDU types: {} \nNumber of RTUs: {}'.format( \
        df.dstIP.unique(), list(df['ASDU_Type'].unique()), len(df['ASDU_addr'].unique())))

def regex_desc(pattern, string):
    return re.search(pattern, string, re.IGNORECASE)


def parse_desc(row_pointId, row_desc):
    label = ''
    if row_pointId.startswith('D'):
        # alarm status, regulator control, expansion emergency control,
        # valve, position, membrane, motor, test
        if regex_desc('(alarm|fault)', row_desc) or row_desc.endswith('Pressure High') or row_desc.endswith(
                'Pressure Low'):
            label = 'alarm'
        elif regex_desc('emergency', row_desc):
            label = 'em_ctrl'
        elif regex_desc('test', row_desc):
            label = 'test'
        else:
            if regex_desc('regulator', row_desc):
                label = 'regulator'
            elif regex_desc('(valve|Pressure|Bypass)', row_desc):
                label = 'valve'
            # elif regex_desc('pressure', row_desc):
            #    label = 'pressure status'
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
    elif row_pointId.startswith('A'):
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


def denormalize(x: float, minval: float, maxval: float) -> float:
    """
    A simple denormalize function
    Parameters:
          x - one input value
          min/max - boundaries
    """
    return (1.0 + x) / 2.0 * (maxval - minval) + minval


def denormalize_val_col(x: pd.Series, minval: float, maxval: float) -> pd.Series:
    """
    Denormalize gas data from 104 traffic in normalized format
    """
    x_denorm = x.apply(denormalize, args=(minval, maxval))
    print('Boundaries: [{}, {}], Before denormalize: {}. After: {}'.format(minval, maxval, x.shape, x_denorm.shape))
    return x_denorm
