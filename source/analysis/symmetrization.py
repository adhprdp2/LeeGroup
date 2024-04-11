import scipy
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime


DATA_PATH = Path(__file__).parents[2] / 'data'


# -------------------------- input parameters -----------------------------
# -------------------------------------------------------------------------
FILE_NAME = 'RvsHvarT_Ch3Sn6_4new_Rxx_Ch2_Rxy.dat'

design_input_parameters = {
    'length': 600,  # um
    'width': 200,  # um
    'thickness': 2  # nm
}

test_input_parameters = {
    'B_max': 14  # T
}

analysis_input_parameters = {

}

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


def parse_file(filepath):
    file_open_time_epoch = 0.
    header_line_count = 0
    for line in open(filepath, 'r'):
        header_line_count += 1
        if line.rstrip().lower() == '[data]': break
        if line.lower().startswith('fileopentime'): file_open_time_epoch = float(line.split(',')[1])

    data_df = pd.read_csv(DATA_PATH / FILE_NAME, skiprows=header_line_count)

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


def extract_performance_parameters(design_parameters, data_df):
    length, width, thickness = design_parameters['length'], design_parameters['width'], design_parameters['thickness']

    #z1 = (symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9][-1]-symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9][0])/(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0][-1]-symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0][0])
    z1 = None
    R_H = z1 * 10000 * thickness * 10**-9 #rh in m(thats why 10^-9)
    n3d = 1/(z1 * (1.602*10**-19) * 10**6) #times 10^6 to get into cm^3 from m^3
    n2d = n3d*(thickness*10**-7) #converting to 2d density. 10^-7 to turn nm to cm

    # R_0 = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9][np.argmin(abs(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0]))]
    R_0 = None
    R_sheet = R_0 * width / length
    resisitivity = (R_sheet * thickness * (10**-9) * width / length)*10**5  # milliOhm cm
    mobility = (1/(resisitivity * (1.602*10**-19) * n3d))*1000  # cm^2/(Vs))

    performance_parameters = {}

    return performance_parameters


def main():
    Bmax = 14 #max field used; important for interpolation fucntion
    thickness = 40*10**-7#thickness in centimeters
    geo = 4.35236 #thickness in centimeters Van der Pauw: pi/ln(2) =4.53236; Hall bar = lengh/Area
    saturation = 6 # % of data used to fit the linear background (i.e. for a 10T scan, 10 would give a fit over 9-10T data range)

    metadata, data_df = parse_file(DATA_PATH / FILE_NAME)

    # data transformation operations
    data_df = column_remap(data_df, file_type=FILE_NAME.split('.')[1])
    data_df = cluster_temperatures(data_df)
    #data_df = smooth_signal(data_df, window_length, 'y')
    #data_df = symmetrize(data_df, 'x', 'y')
    #data_df = antisymmetrize(data_df, 'x', 'y')

    #performance_parameters = extract_performance_parameters(data_df)

    print(data_df[['Temperature (K)', 'temperature_cluster', 'Magnetic Field (Oe)']])
    print(data_df.columns)

main()
