import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import odr
import plotly.express as px
import plotly.graph_objects as go

def read_lvm(path, date_i=9, time_i=10, header_i=22):
    '''
    input = lvm file from labview.
    return dataframe with datetime info of load cells data.
    '''
    with open(path, 'r') as f:
        lines = f.readlines()
    date_time = pd.to_datetime(lines[date_i].strip().split('\t')[1] + ' ' + lines[time_i].strip().split('\t')[1])
    data = []
    for line in lines[header_i + 1:]:
        try:
            data.append(pd.to_numeric(line.strip().split('\t')))
        except:
            pass
    df = pd.DataFrame(data, columns=lines[header_i].strip().split('\t')[:len(data[0])])
    df['datetime'] = pd.to_timedelta(df.X_Value, unit='s') + date_time
    return df


def load2zeros(df, start_i=0, duration_s=90):
    dt_ini = df.datetime[start_i]
    dt_end = dt_ini + pd.to_timedelta(duration_s, unit='s')
    l1, l2 = df.loc[(df.datetime >= dt_ini) & (df.datetime <= dt_end), ['MS-3k-S_Loadcell (Resampled)',
                                                                        'Airtech 3k ZLoad-CH2 (Resampled)']].mean().values
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
                # t_ini = df_t[df_t['event'] == 'START']['system time'].tolist()[0] - 20 * 60 * 60
                # df['time'] = df['t (s)'] + t_ini
                df_t['system time text'] = pd.to_datetime(df_t['system time text'])
                t_ini = df_t['system time text'][0]
                df['time'] = pd.to_timedelta(df['t (s)'], unit='s') + t_ini
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


def phyphox_join(folders, interpolation=True, all_data=True, out_path=None):
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
        key = 'Top' if 'top' in folder_name else 'Bottom' if 'bot' in folder_name else folder_name.split('_')[2]
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

    if all_data:
        return df
    else:
        return df[['time'] + [col for col in df.columns if 'Plane Inclination' in col]]


def printMatch(lista: list, key='2_1'):
    """
    :param lista: list of paths to look up the key
    :return: print matched elements to the key
    """
    for i, l in enumerate(lista):
        if key in l:
            print(i, l)


def loadDIC(path, df_DIC_info, sampleID: str, camera='r', by='lastFrame'):
    """
    to read a csv file and get datetime based on lastFrame or duration
    :param by: microsecond of the last frame: lastFrame or duration
    :param path: DIC results csv path
    :param df_DIC_info: dataframe of info recollected during the tests
    :param sampleID: bamboo sample e.g. '2_1'
    :param camera: 'r' or 'l' for right and left cameras
    :return: dataframe with DIC results and datetime
    """
    df = pd.read_csv(path)
    sample_data = df_DIC_info[df_DIC_info.sampleID == sampleID]
    found = False
    for i in sample_data.index:
        if camera in sample_data.camera[i]:
            found = True
            end_time = pd.to_datetime(sample_data.datetime[i]) - pd.to_timedelta(4, 'hour')
            break
    if not found:
        raise Exception('Could no find camera')

    end_msec = df_DIC_info[by].values[i]
    start_time = end_time - pd.to_timedelta(end_msec / 10 ** 6, 'sec')
    df['datetime'] = start_time + pd.to_timedelta(df.microSeconds / 10 ** 6, 'sec')
    return df

def ODR_results(df, title = None): # label_x = '$\gamma$', label_y = '$\tau$ (ksi)'
    """
    Regression using Orthogonal Distance Regression method.
    :param df:  dataframe with x and y info. x: strain, deformation, etc. y: stress,
    shear stress, torque, etc.
    :param title: optional.
    :return: plotly figure and a and b constants of the equation y = ax + b
    """
    cols = df.columns
    x, y = df[cols[0]], df[cols[1]]
    data = odr.Data(x, y)
    odr_obj = odr.ODR(data, odr.unilinear)
    output = odr_obj.run()
    a, b = output.beta

    fig1 = px.scatter(df, x = cols[0], y = cols[1], opacity = 0.65)# , labels = {'x': cols[0], 'y': cols[1]}
    fig2 = px.line(x = df[cols[0]], y = df[cols[0]]*a+b)
    fig2.update_traces(line=dict(color = 'darkgray', width = 1, dash = 'dash'))
    yl = cols[1].replace('$', '')
    xl = cols[0].replace('$', '')
    fig2.add_annotation(x = x.min()*1.1, y = y.min()*.4, text = f'${yl} = {a}{xl}+{b}$')

    fig3 = go.Figure(data = fig1.data + fig2.data)
    fig3.show()
    return fig3, a, b


def plotRing(df_fib, img_path):
    """
    generate a plotly figure of the fiber density behavior with the image of the ring
    :param df_fib: rings and wedges with fiber density results
    :param img_path: path of the scanned ring
    :return: plotly figure
    """
    return 'in process'
