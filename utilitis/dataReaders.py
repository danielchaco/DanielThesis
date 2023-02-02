import pandas as pd
import os

def read_lvm(path, date_i = 9, time_i = 10, header_i = 22):
    '''
    input = lvm file from labview.
    return dataframe with datetime info of load cells data.
    '''
    with open(path,'r') as f:
        lines = f.readlines()
    date_time = pd.to_datetime(lines[date_i].strip().split('\t')[1] +' ' + lines[time_i].strip().split('\t')[1])
    data = []
    for line in lines[header_i+1:]:
        try:
            data.append(pd.to_numeric(line.strip().split('\t')))
        except:
            pass
    df = pd.DataFrame(data,columns=lines[header_i].strip().split('\t')[:len(data[0])])
    df['datetime'] = pd.to_timedelta(df.X_Value,unit='s') + date_time
    return df

def load2zeros(df, start_i = 0, duration_s = 90):
    dt_ini = df.datetime[start_i]
    dt_end = dt_ini + pd.to_timedelta(duration_s,unit='s')
    l1,l2 = df.loc[(df.datetime>=dt_ini)&(df.datetime<=dt_end),['MS-3k-S_Loadcell (Resampled)','Airtech 3k ZLoad-CH2 (Resampled)']].mean().values
    df['MS-3k-S_Loadcell (Resampled)'] = df['MS-3k-S_Loadcell (Resampled)'] - l1
    df['Airtech 3k ZLoad-CH2 (Resampled)'] = df['Airtech 3k ZLoad-CH2 (Resampled)'] - l2
    return df


def file_paths(folder):
    '''
    returns list of paths of all files contained in a main folder by walking
    '''
    file_list = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            file_list.append(os.path.join(root, filename))
    return file_list


def read_lines(path, get_lines=True):
    '''
    *.lvm files from labview (load cells data)
    '''
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        print(line.split('\n')[0])
    if get_lines:
        return lines


def merge_phyphox_data(phyphox_data):
    '''
    from csv phyphox folder, independently sensor files are compiled in a dataframe
    param phyphox_data: list of sensor file paths or container folder path.
    '''
    if type(phyphox_data) == list:
        data_file_paths = phyphox_data
    else:
        data_file_paths = file_paths(phyphox_data)
    DF, dnames = [], ['Flat', 'Plane', 'Side', 'Upright']
    for file_path in data_file_paths[0:4]:
        bname = os.path.basename(file_path).split('.')[0]
        if bname in dnames:
            df = pd.read_csv(file_path)
            df.columns = [col if col == 't (s)' else bname + ' ' + col for col in df.columns]
            DF.append(df)
    if len(DF) > 0:
        df = pd.concat(DF, axis=1)
        df = df.T.drop_duplicates().T
    else:
        print('no dataframes')
    try:
        found = False
        for path in data_file_paths:
            if 'time' in path:
                found = True
                df_t = pd.read_csv(path)
                t_ini = df_t[df_t['event'] == 'START']['system time'].tolist()[0] - 20 * 60 * 60
                df['time'] = df['t (s)'] + t_ini
                return df
        if not found:
            print('metadata file time.csv, not found.')
            return df
    except BaseException as err:
        print(f"Unexpected {err}, {type(err)}")
        return df


def plot_phyphox_data(phyphox_folder):
    df = merge_phyphox_data(file_paths(phyphox_folder))
    ax = df.plot('t (s)', ['Flat Tilt up/down (deg)', 'Flat Tilt left/right (deg)',
                           'Plane Inclination (deg)', 'Plane Rotation (deg)',
                           'Side Tilt up/down (deg)', 'Side Tilt left/right (deg)',
                           'Upright Tilt up/down (deg)'], figsize=(20, 10), grid=True)
    plt.title(phyphox_folder.split('\\')[-1])


def phyphox_join(folders, interpolation=True, out_path=None):
    '''
    returns a dataframe with information of several phyphox folders, recorded at
    the simultaneusly.

    param: folders: list with paths of pyphox folders with bottom or top label,
    if bottom or top is not in the name, they will be numerated as 0,1,2,...
    param: interpolation. If interpolation is set to True, empty rows will be filled
    by interpolation otherwise they will remain empty.
    param: out_path, if there is a value, dataframe will be recorded as a csv file
    in the provided path.
    '''
    DF = []
    for folder in folders:
        folder_name = os.path.basename(folder)
        key = 'top' if 'top' in folder_name else 'bottom' if 'bottom' in folder_name else folder_name.split('_')[2]
        df = merge_phyphox_data(folder)
        cols = []
        for col in df.columns:
            if col not in ['t (s)', 'time']:
                cols.append(key + ' ' + col)
            else:
                cols.append(col)
        df.columns = cols
        DF.append(df)
    df = pd.concat(DF).sort_values(by='time')
    df.reset_index(inplace=True, drop=True)

    if interpolation:
        for col in df.columns:
            if col not in ['t (s)', 'time']:
                df[col] = df[col].interpolate()

    if out_path:
        df.to_csv(out_path, index=False)

    return df