import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import odr
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


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
        for path in data_file_paths:
            if 'time' in path:
                df_t = pd.read_csv(path)
                # t_ini = df_t[df_t['event'] == 'START']['system time'].tolist()[0] - 20 * 60 * 60
                # df['time'] = df['t (s)'] + t_ini
                df_t['system time text'] = pd.to_datetime(df_t['system time text'])
                t_ini = df_t['system time text'][0]
                df['datetime'] = pd.to_timedelta(df['t (s)'], unit='s') + t_ini
                df.datetime = df.datetime.dt.tz_localize(None)
                return df
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
        try:
            if key in l:
                print(i, l)
        except:
            print(i, l)
            pass


def loadDIC(path, df_DIC_info, sampleID: str, camera='r', by='lastFrame'):
    """
    to read a csv file and get datetime based on lastFrame or duration
    :param by: microsecond of the last frame: lastFrame or duration
    :param path: DIC results csv path
    :param df_DIC_info: dataframe of info recollected during the tests
    :param sampleID: bamboo sample e.g. '2_1'
    :param camera: 'c', 'r' or 'l' for color, right and left cameras
    :return: dataframe with DIC results and datetime
    """
    df = pd.read_csv(path)
    sample_data = df_DIC_info[df_DIC_info.sampleID == sampleID]
    found = False
    for i in sample_data.index:
        if camera in sample_data.camera[i]:
            found = True
            end_time = pd.to_datetime(sample_data.datetime[i]).tz_convert('America/Puerto_Rico')
            break
    if not found:
        raise Exception('Could no find camera')

    end_msec = df_DIC_info[by].values[i]
    start_time = end_time - pd.to_timedelta(end_msec / 10 ** 6, 'sec')
    df['datetime'] = start_time + pd.to_timedelta(df.microSeconds / 10 ** 6, 'sec')
    df.datetime = df.datetime.dt.tz_localize(None)
    return df


def ODR_results(df, title=None):  # label_x = '$\gamma$', label_y = '$\tau$ (ksi)'
    """
    Regression using Orthogonal Distance Regression method.
    :param df:  dataframe with x and y info. x: strain, deformation, etc. y: stress,
    shear stress, torque, etc.
    :param title: optional.
    :return: plotly figure and a and b constants of the equation y = ax + b
    """
    cols = df.columns
    x, y = df[cols[0]].to_numpy(), df[cols[1]].to_numpy()
    data = odr.Data(x, y)
    odr_obj = odr.ODR(data, odr.unilinear)
    output = odr_obj.run()
    a, b = output.beta

    fig1 = px.scatter(df, x=cols[0], y=cols[1], opacity=0.65)  # , labels = {'x': cols[0], 'y': cols[1]}
    fig2 = px.line(x=df[cols[0]], y=df[cols[0]] * a + b)
    fig2.update_traces(line=dict(color='darkgray', width=1, dash='dash'))
    yl = cols[1].replace('$', '')
    xl = cols[0].replace('$', '')
    fig2.add_annotation(x=x.min() * 1.1, y=y.min() * .4, text=f'${yl} = {a}{xl}+{b}$')

    fig3 = go.Figure(data=fig1.data + fig2.data)
    fig3.show()
    return fig3, a, b


def plotRing(df_fib, img_path, resize=200, figureOpt = 1):
    """
    generate a plotly figure of the fiber density behavior with the image of the ring
    :param figureOpt: 1 for clasic color lines, 2 for wide gray traces, 3 wide color traces.
    :param resize: the new size of the scan which is huge.
    :param df_fib: rings and wedges with fiber density results
    :param img_path: path of the scanned ring
    :return: plotly figure
    """
    df_fib.rename(columns={'density': 'Density (%)', 'wedge': 'Wedge'}, inplace=True)
    df_fib['t (%)'] = df_fib.ring / df_fib.ring.max()
    df_fib = df_fib[~df_fib.ring.isin([df_fib.ring.min(), df_fib.ring.max()])]

    fig = px.line(df_fib, x="t (%)", y="Density (%)", color='Wedge', template='plotly_white')

    if figureOpt == 1:
        pass
    elif figureOpt == 2:
        fig.update_traces(line=dict(color='darkgray', width=10), opacity=0.5)
    elif figureOpt == 3:
        fig.update_traces(line=dict(width=10), opacity=0.5)
    else:
        raise Exception('Enter a valid figureOpt: 1, 2, 3')
    img = Image.open(img_path)
    img_size = img.size
    r = resize / img_size[0]
    img = img.resize((resize, int(img_size[1] * r)))
    fig.add_layout_image(
        dict(
            source=img,
            x=0.95,
            y=0.1,
        ))
    fig.update_layout_images(dict(
        xref="paper",
        yref="paper",
        sizex=0.30,
        sizey=0.30,
        xanchor="right",
        yanchor="bottom"
    ))

    fig.update_layout(font_family='Times New Roman', margin=dict(l=5, r=10, t=10, b=5))
    return fig


def phy_bot_top(folders, all_data=False):
    """
    helps to select which is bottom or top from folders
    :param all_data: if true returns all sensors' data, else returns just inclination data
    :param folders: folders of bottom and top results of phyphox of same test
    :return: dataframes of bottom and top readings
    """
    count = 0
    for folder in folders:
        if 'top' in os.path.basename(folder):
            phy_top = merge_phyphox_data(folder)
            count += 1
        if 'bot' in os.path.basename(folder):
            phy_bot = merge_phyphox_data(folder)
            count += 1
    if count != 2:
        raise Exception(f'Could not find bottom or top paths in {folders}')
    else:
        if all_data:
            return phy_bot, phy_top
        else:
            return phy_bot[['datetime'] + [col for col in phy_bot.columns if 'Plane Inclination' in col]], phy_top[
                ['datetime'] + [col for col in phy_top.columns if 'Plane Inclination' in col]]


def max_ortho_dist_index(df):
    """
    measure the perpendicular distance of a line between first and end points of a list of points
    It is assumed that datetime and Plane Inclination are in df.
    :param df: dataframe with x and y points
    :return: the location where the maximum ortho distance is
    """
    for col in df.columns:
        if 'Plane Inclination' in col:
            break
    s = df.datetime.view('int64') / 10 ** 9  # .view, .astype
    s.rename('t', inplace=True)
    df = pd.concat([df, s], axis=1)
    idx = df.index
    a, b = idx[0], idx[-1]
    Q = df.loc[a:b, ['t', col]].to_numpy()
    A = df.loc[a, ['t', col]].to_numpy()
    B = df.loc[b, ['t', col]].to_numpy()
    v = B - A
    AQ = Q - A
    d = np.abs(np.cross(v, AQ) / np.linalg.norm(v))
    return idx[np.where(d == d.max())[0][0]]


def failure_times(df_load, df_dic, phy_bot, phy_top, failure_time_aprox: None):
    """
    define the time where the sample reaches the max load
    :param df_load: dataframe with load cell info
    :param failure_time_aprox: if None, it will look at points before the 60% of max time
    :return: datetime when failure happened in the order: load, dic, phy_bot, phy_top
    """
    df_load = df_load.copy()
    df_load['MS_diff'] = df_load['MS-3k-S_Loadcell (Resampled)'].diff().abs()
    df_load['Air_diff'] = df_load['Airtech 3k ZLoad-CH2 (Resampled)'].diff().abs()
    failure_time_aprox = failure_time_aprox if failure_time_aprox else df_load.datetime[int(len(df_load) * .6)]
    max1 = df_load[df_load.datetime < failure_time_aprox]['MS_diff'].max()
    max2 = df_load[df_load.datetime < failure_time_aprox]['Air_diff'].max()
    failure1 = df_load[df_load['MS_diff'] == max1].datetime.values[0]
    failure2 = df_load[df_load['Air_diff'] == max2].datetime.values[0]

    time_failure_load = min(failure1, failure2)

    col = 'mean' if len(df_dic.columns) > 2 else df_dic.columns[1]
    df_tem = df_dic[(df_dic.datetime >= time_failure_load - pd.to_timedelta(10, 's')) & (
            df_dic.datetime <= time_failure_load + pd.to_timedelta(10, 's'))]
    diff = df_tem[col].diff().abs()
    diffname = diff.name + '_diff'
    diff.rename(diffname, inplace=True)
    df_tem = pd.concat([df_tem, diff], axis=1)

    time_failure_dic = df_tem[df_tem[diffname] == df_tem[diffname].max()].datetime.values[0]

    phy_bot_slice = phy_bot[(phy_bot.datetime >= time_failure_load - pd.to_timedelta(20, 's')) & (
            phy_bot.datetime <= time_failure_load + pd.to_timedelta(20, 's'))]
    time_failure_phy_bot = phy_bot.datetime[max_ortho_dist_index(phy_bot_slice)]

    phy_top_slice = phy_top[(phy_top.datetime >= time_failure_load - pd.to_timedelta(20, 's')) & (
            phy_top.datetime <= time_failure_load + pd.to_timedelta(20, 's'))]
    time_failure_phy_top = phy_top.datetime[max_ortho_dist_index(phy_top_slice)]

    return time_failure_load, time_failure_dic, time_failure_phy_bot, time_failure_phy_top


def get_seconds(t1, t2):
    """
    difference of time between t1 and t2. t2 >= t1
    :param t1: datetime or timestamp of first time
    :param t2: datetime or timestamp of first time
    :return: time in seconds t2 - t1
    """
    try:
        return (t1 - t2).total_seconds()
    except:
        return (t1 - t2).item() / 10 ** 9


def mergeData(df_load, phy_bot, phy_top, df_dic):
    """
    marge load cell, phones and DIC data, after matching failure point
    :param df_load:
    :param phy_bot:
    :param phy_top:
    :param df_dic:
    :return: dataframe with all data
    """
    df_load, df_dic = df_load.copy(), df_dic.copy()
    phy_bot = phy_bot[['datetime', 'Plane Inclination (deg)']].rename(
        columns={'Plane Inclination (deg)': 'Bottom Plane Inclination (deg)'})
    phy_top = phy_top[['datetime', 'Plane Inclination (deg)']].rename(
        columns={'Plane Inclination (deg)': 'Top Plane Inclination (deg)'})

    for df in df_load, phy_bot, phy_top, df_dic:
        df.set_index('datetime', inplace=True)

    phy_bot = phy_bot.loc[~phy_bot.index.duplicated(keep='first')]
    phy_top = phy_top.loc[~phy_top.index.duplicated(keep='first')]

    df_phones = pd.concat([phy_bot, phy_top], axis=1).sort_values(by='datetime')
    df_phones.interpolate(inplace=True)
    DF = [
        df_load[['MS-3k-S_Loadcell (Resampled)', 'Airtech 3k ZLoad-CH2 (Resampled)']],
        df_phones,
        df_dic[['min', 'max', 'mean', 'median']]
    ]
    df = pd.concat(DF, axis=1).sort_values(by='datetime')
    df.reset_index(inplace=True)

    for col in df.columns:
        if col not in ['datetime', 'Bottom Plane Inclination (deg)', 'Top Plane Inclination (deg)', 'min', 'max',
                       'mean', 'median']:
            df[col] = df[col].interpolate()
    return df
