import yaml
import scipy
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime


# -------------------------- input parameters -----------------------------
# -------------------------------------------------------------------------
ANALYSIS_TYPE = 'hall_bar'

FILE_NAME = 'hall-bar__wafer9999-chip1-hbar2__2023-10-10_23-34-05.dat'
DESIGN_NAME = 'default_hall_bar_device'

TEST_INPUT_PARAMETERS = {
    'B_max': 14  # T
}

ANALYSIS_INPUT_PARAMETERS = {
    'window_length': 10
}

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

DATA_PATH = Path(__file__).parents[2] / 'data'

SOURCE_PATH = DATA_PATH / 'measurement' / 'raw' / ANALYSIS_TYPE
TARGET_PATH = DATA_PATH / 'measurement' / 'processed' / ANALYSIS_TYPE

DESIGN_FILE = 'design_parameters.yml'

def parse_file(filepath):
    file_open_time_epoch = 0.
    header_line_count = 0
    for line in open(filepath, 'r'):
        header_line_count += 1
        if line.rstrip().lower() == '[data]': break
        if line.lower().startswith('fileopentime'): file_open_time_epoch = float(line.split(',')[1])

    data_df = pd.read_csv(filepath, skiprows=header_line_count)

    data_df['Time Stamp (sec)'] = (pd.to_datetime(data_df['Time Stamp (sec)'], unit='s'))

    metadata = {
        'file_open_time': datetime.fromtimestamp(file_open_time_epoch)
    }

    return metadata, data_df


def column_remap(data_df, file_type=None):

    if file_type == '.dat':
        pass
    if file_type == '.csv':
        pass

    Cols = 4 #number of data columns per B field(include B field column)
    #below is the column number of each of the data points beginning at position zero
    #this lets the code know which columns are what
    Tpos = 0
    Bpos = 1
    Rxx1pos = 2
    Rxx2pos =2
    Rxypos =3 #set Rxypos = 99 if there is no Rxy data
    savedFileName = 'AnjSn645.csv' #a csv file with this name will be saved. two files of the exact same name cannot exist in the folder
    n2dFileName = 'n2dDataoutout.csv'

    #are you using Ryy data in place of Rxx2?
    #if yes, put Ryypos = 101
    #if no, put Ryypos = 201
    Ryypos = 201
    return data_df


def cluster_temperatures(data_df):
    threshold = 0.050 # can resolve cluster with temperature variation up to 50 mK
    data_df['temperature_cluster'] = data_df.sort_values(by=['Temperature (K)'])['Temperature (K)'].diff().gt(threshold).cumsum().add(1)

    # TODO: Add column for inference of setpoint temperature

    return data_df


def smooth_signal(data_df, window_length, signal_column_name):
    y = data_df[signal_column_name]
    smoothed_signal = scipy.signal.savgol_filter(y, window_length, polyorder=1)
    data_df[signal_column_name + '_smoothed'] = smoothed_signal
    return data_df


def symmetrize(data_df, x_column_name, y_column_name):
    # up sweep vs. down sweep
    # ...
    x = data_df[x_column_name]
    y = data_df[y_column_name]

    upper_bound = np.min([np.abs(np.max(x)), np.abs(np.min(x))])
    x_symm = np.linspace(0, upper_bound, len(x))
    fit = scipy.interpolate.interp1d(x, y, 'linear')
    y_symm = (fit(+1 * x_symm) + fit(-1 * x_symm))/2

    data_df[x_column_name + '_symmetrized'] = x_symm
    data_df[y_column_name + '_symmetrized'] = y_symm

    return data_df


def antisymmetrize(data_df, x_column_name, y_column_name):
    # up sweep vs. down sweep
    # ...
    x = data_df[x_column_name]
    y = data_df[y_column_name]

    upper_bound = np.min([np.abs(np.max(x)), np.abs(np.min(x))])
    x_antisymm = np.linspace(0, upper_bound, len(x))
    fit = scipy.interpolate.interp1d(x, y, 'linear')
    y_antisymm = (fit(+1 * x_antisymm) - fit(-1 * x_antisymm))/2

    data_df[x_column_name + '_antisymmetrized'] = x_antisymm
    data_df[y_column_name + '_antisymmetrized'] = y_antisymm

    return data_df


def extract_performance_parameters(device_design, data_df):
    length, width, thickness = device_design['length'], device_design['width'], device_design['thickness']

    #z1 = (symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9][-1]-symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9][0])/(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0][-1]-symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0][0])
    z1 = 1

    r_h = z1 * 10000 * thickness * 10**-9 #rh in m(thats why 10^-9)
    carrier_density_3d = 1/(z1 * (1.602*10**-19) * 10**6) #times 10^6 to get into cm^3 from m^3
    carrier_density_2d = carrier_density_3d*(thickness*10**-7) #converting to 2d density. 10^-7 to turn nm to cm

    # R_0 = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9][np.argmin(abs(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0]))]
    r_0 = 1

    r_sheet = r_0 * width / length
    resisitivity = (r_sheet * thickness * (10**-9) * width / length)*10**5  # milliOhm cm
    hall_mobility = (1/(resisitivity * (1.602*10**-19) * carrier_density_3d))*1000  # cm^2/(Vs))

    performance_parameters = {
        'r_h': r_h,
        'r_0': r_0,
        'r_sheet': r_sheet,
        'resistivity': resisitivity,
        'carrier_density_2d': carrier_density_2d,
        'hall_mobility': hall_mobility
    }

    return performance_parameters


def main():
    metadata, data_df = parse_file(SOURCE_PATH / FILE_NAME)
    device_design  = yaml.load(open(DATA_PATH / 'design' / DESIGN_FILE, 'r'), Loader=yaml.Loader)[DESIGN_NAME]  # load before lengthy transforms in case of error

    window_length = ANALYSIS_INPUT_PARAMETERS['window_length']

    # --- data transformation operations ---
    try:
        data_df = column_remap(data_df, file_type=FILE_NAME.split('.')[1])
        data_df = cluster_temperatures(data_df)
        data_df = smooth_signal(data_df, window_length)
        data_df = symmetrize(data_df)
        data_df = antisymmetrize(data_df)
    except:
        pass  # temporarily skip code that isn't yet finished

    # compute collateral for reporting inputs
    #   1.) desired params - resistivity, Hall mobility, etc. - from data
    #   2.) processed raw data for future graphs, checks, or other further analysis
    performance_parameters = extract_performance_parameters(device_design, data_df)

    print(data_df[['Temperature (K)', 'temperature_cluster', 'Magnetic Field (Oe)']])
    print(data_df.columns)
    print(performance_parameters)

main()
