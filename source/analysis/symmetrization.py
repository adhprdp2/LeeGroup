import pandas as pd

from pathlib import Path
from datetime import datetime

DATA_PATH = Path(__file__).parents[2] / 'data'
FILE_NAME = 'RvsHvarT_Ch3Sn6_4new_Rxx_Ch2_Rxy.dat'


def parse_file(filepath):
    file_open_time_epoch = 0.
    header_line_count = 0
    for line in open(filepath, 'r'):
        header_line_count += 1
        if line.rstrip().lower() == '[data]': break
        if line.lower().startswith('fileopentime'): file_open_time_epoch = float(line.split(',')[1])

    data_df = pd.read_csv(DATA_PATH / FILE_NAME, skiprows=header_line_count)

    metadata = {
        'file_open_time': datetime.fromtimestamp(file_open_time_epoch)
    }

    return metadata, data_df


def main():
    metadata, data_df = parse_file(DATA_PATH / FILE_NAME)

    formatted_time = metadata['file_open_time'].strftime('%Y-%m-%d %H:%M:%S')
    print(formatted_time)
    print(data_df)

main()
