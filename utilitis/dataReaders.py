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
                start, end = None, None
                for i in df_t.index:
                    if 'START' in df_t.event[i]:
                        start = i
                    elif 'PAUSE' in df_t.event[i]:
                        end = i
                    if start is not None and end is not None:
                        frm = df.index[df['t (s)'] > df_t['experiment time'][start]]
                        to = df.index[df['t (s)'] <= df_t['experiment time'][end]]
                        df.loc[frm[0]:to[-1], 'datetime'] = df_t['system time text'][start]
                        df.loc[frm[0]:to[-1], 't'] = df.loc[frm[0]:to[-1], 't (s)'] - df_t['experiment time'][start]
                        start, end = None, None
                df['datetime'] = pd.to_datetime(df['datetime']) + pd.to_timedelta(df['t'], unit='s')
                df.drop(columns=['t'], inplace=True)
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


def ODR_results(df, start_time, end_time, fig_show=True, title=None, colors=None, min_max=(0.15, 0.5), tau_bt = None):
    """
    Regression using Orthogonal Distance Regression method.
    resources: https://github.com/plotly/plotly.py/issues/2345#issuecomment-858396014
    :param df:  dataframe with merged data.
    :param title: optional.
    :return: plotly figure and a and b constants of the equation y = ax + b
    """
    start_time = start_time if type(start_time) != str else pd.to_datetime(start_time)
    end_time = end_time if type(end_time) != str else pd.to_datetime(end_time)
    cols = [col for col in df.columns if col in ['$\gamma_{DIC}$', '$\gamma_{phones}$']]
    if not colors:
        colors = px.colors.qualitative.Pastel
    df = df.loc[(df.datetime >= start_time) & (df.datetime <= end_time), ['$\\tau (ksi)$'] + cols]
    df.reset_index(drop=True, inplace=True)
    max_tau = df['$\\tau (ksi)$'].max()
    df.rename(columns={'$\\tau (ksi)$':'$\\tau_{23} (ksi)$'},inplace=True)
    fig = px.scatter(df, cols, '$\\tau_{23} (ksi)$', color_discrete_sequence=colors, opacity=1)
    fig.update_traces(marker=dict(size=1))  # ,line=dict(width=2,color=colors)
    fig.update_layout(template='simple_white', font_family='Times New Roman', margin=dict(l=5, r=10, t=10, b=0), #plotly_white
                      xaxis=dict(title='$\gamma_{23}$'), legend_title="", yaxis_range=[0, max_tau*1.1],
                      legend=dict(yanchor="bottom", y=0, xanchor="right", x=1))  # height=400, width=900,
    # trendlines
    frm = int(len(df) * min_max[0])
    to = int(len(df) * min_max[1])
    G = []
    for i, col in enumerate(cols):
        df_c = df[~pd.isnull(df[col])]
        x = df_c.loc[frm:to, col].to_numpy()
        y = df_c.loc[frm:to, '$\\tau_{23} (ksi)$'].to_numpy()
        data = odr.Data(x, y)
        odr_obj = odr.ODR(data, odr.unilinear)
        output = odr_obj.run()
        a, b = output.beta
        x = np.linspace(0, (max_tau - b) / a, 100)
        fig.add_trace(go.Scattergl(x=x, y=x * a + b, mode='lines', name=f'$\\tau={b:.3f}+{a:.3f}\gamma$',
                                   line={'width': 1, 'dash': ['dash', 'dashdot'][i], 'color': px.colors.sequential.gray[
                                       i * 4]}))  # ['dash', 'dashdot', 'dot', 'longdash', 'longdashdot','solid']
        G.append(a)
    if tau_bt:
        x_min = np.min(df[cols].min().values)
        x_max = np.max(df[cols].max().values)
        fig.add_trace(go.Scattergl(x=[x_min,x_max],y = [tau_bt,tau_bt], mode = 'lines', name = f'$\\tau = {tau_bt} ksi$',
                                   line={'width': 1, 'dash': 'dash', 'color':'red'}))
    if title:
        fig.update_layout(title=title)
    if fig_show:
        fig.show()
    return fig, G


def plotRing(df_fib, img_path, resize=200, figureOpt=1):
    """
    generate a plotly figure of the fiber density behavior with the image of the ring
    :param figureOpt: 1 for clasic color lines, 2 for wide gray traces, 3 wide color traces.
    :param resize: the new size of the scan which is huge.
    :param df_fib: rings and wedges with fiber density results
    :param img_path: path of the scanned ring
    :return: plotly figure
    """
    df_fib['ddiff'] = [np.nan] * len(df_fib)
    for w in df_fib.wedge.unique():
        for r in df_fib.ring.unique():
            try:
                df_fib.loc[(df_fib.ring == r) & (df_fib.wedge == w), 'ddiff'] = \
                    df_fib.loc[(df_fib.ring == r) & (df_fib.wedge == w), 'density'].values[0] - \
                    df_fib.loc[(df_fib.ring == r - 1) & (df_fib.wedge == w), 'density'].values[0]
            except:
                pass
    df_fib.ddiff = df_fib.ddiff.abs()
    df_fib.loc[(df_fib.ring > df_fib.ring.max() / 2) & (df_fib.ddiff > .18), 'density'] = np.nan
    for i in range(1, int(df_fib.ring.max() / 2) + 1):
        w_list = df_fib[(df_fib.ring == i + 1) & (df_fib.ddiff > .18)].wedge.unique()
        if len(w_list) > 0:
            df_fib.loc[(df_fib.ring == i) & df_fib.wedge.isin(w_list), 'density'] = np.nan

    df_fib.density = df_fib.density * 100
    df_fib.rename(columns={'density': 'Density (%)', 'wedge': 'Wedge'}, inplace=True)
    df_fib['t (%)'] = df_fib.ring / df_fib.ring.max() * 100
    df_fib.dropna(subset=['Wedge', 'Density (%)'], inplace=True)

    fig = px.line(df_fib, x="t (%)", y="Density (%)", color='Wedge', template='plotly_white')

    if figureOpt == 1:
        pass
    elif figureOpt == 2:
        fig.update_traces(line=dict(color='darkgray', width=13), opacity=0.5)
        fig.update_layout(showlegend=False)
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
            x=0.05,
            y=0.95,
        ))
    fig.update_layout_images(dict(
        xref="paper",
        yref="paper",
        sizex=0.30,
        sizey=0.30,
        xanchor="left",
        yanchor="top"
    ))
    fig.update_layout(yaxis_range=[0, 100], xaxis_range=[0, 100], yaxis_ticksuffix="%", xaxis_ticksuffix="%",
                      font_family='Times New Roman', margin=dict(l=5, r=10, t=10, b=5))
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


def failure_times(df_load, df_dic=None, phy_bot=None, phy_top=None, failure_time_aprox=None):
    """
    define the time where the sample reaches the max load
    :param phy_top:
    :param phy_bot:
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

    if df_dic is not None:
        col = 'mean' if len(df_dic.columns) > 2 else df_dic.columns[1]
        df_tem = df_dic[(df_dic.datetime >= time_failure_load - pd.to_timedelta(10, 's')) & (
                df_dic.datetime <= time_failure_load + pd.to_timedelta(10, 's'))]
        diff = df_tem[col].diff().abs()
        diffname = diff.name + '_diff'
        diff.rename(diffname, inplace=True)
        df_tem = pd.concat([df_tem, diff], axis=1)

        time_failure_dic = df_tem[df_tem[diffname] == df_tem[diffname].max()].datetime.values[0]
    else:
        time_failure_dic = None

    if phy_bot is not None:
        phy_bot_slice = phy_bot[(phy_bot.datetime >= time_failure_load - pd.to_timedelta(20, 's')) & (
                phy_bot.datetime <= time_failure_load + pd.to_timedelta(20, 's'))]
        time_failure_phy_bot = phy_bot.datetime[max_ortho_dist_index(phy_bot_slice)]
    else:
        time_failure_phy_bot = None

    if phy_top is not None:
        phy_top_slice = phy_top[(phy_top.datetime >= time_failure_load - pd.to_timedelta(20, 's')) & (
                phy_top.datetime <= time_failure_load + pd.to_timedelta(20, 's'))]
        time_failure_phy_top = phy_top.datetime[max_ortho_dist_index(phy_top_slice)]
    else:
        time_failure_phy_top = None

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


def mergeData(df_load, df_dic=None, phy_bot=None, phy_top=None):
    """
    marge load cell, phones and DIC data, after matching failure point
    :param df_load:
    :param phy_bot:
    :param phy_top:
    :param df_dic:
    :return: dataframe with all data
    """
    # df_load, df_dic = df_load.copy(), df_dic.copy()
    if phy_bot is not None:
        phy_bot = phy_bot[['datetime', 'Plane Inclination (deg)']].rename(
            columns={'Plane Inclination (deg)': 'Bottom Plane Inclination (deg)'})
    if phy_top is not None:
        phy_top = phy_top[['datetime', 'Plane Inclination (deg)']].rename(
            columns={'Plane Inclination (deg)': 'Top Plane Inclination (deg)'})

    for df in df_load, phy_bot, phy_top, df_dic:
        try:
            df.set_index('datetime', inplace=True)
        except:
            pass

    if phy_bot is not None:
        phy_bot = phy_bot.loc[~phy_bot.index.duplicated(keep='first')]
    if phy_top is not None:
        phy_top = phy_top.loc[~phy_top.index.duplicated(keep='first')]

    if phy_bot is not None and phy_top is not None:
        df_phones = pd.concat([phy_bot, phy_top], axis=1).sort_values(by='datetime')
        df_phones.interpolate(inplace=True)
        df_phones = [df_phones]
    else:
        df_phones = []

    DF = [
        df_load[['MS-3k-S_Loadcell (Resampled)', 'Airtech 3k ZLoad-CH2 (Resampled)']],
        df_dic[['min', 'max', 'mean', 'median']]
    ] if df_dic is not None else [
        df_load[['MS-3k-S_Loadcell (Resampled)', 'Airtech 3k ZLoad-CH2 (Resampled)']]
    ]

    DF += df_phones
    df = pd.concat(DF, axis=1).sort_values(by='datetime')
    df.reset_index(inplace=True)

    for col in df.columns:
        if col not in ['datetime', 'Bottom Plane Inclination (deg)', 'Top Plane Inclination (deg)', 'min', 'max',
                       'mean', 'median']:
            df[col] = df[col].interpolate()
    return df


def getCProp(df_ring_prop, top_i, bot_i, transverse=True, return_df=False):
    df = df_ring_prop.loc[df_ring_prop.index.isin([top_i, bot_i]), df_ring_prop.columns].rename(
        index={top_i: 'Top', bot_i: 'Bottom'})  # reset_index()
    df['Fiber Density'] = df['Fiber Density'] * 100
    df = df[['outer_diameter_in', 'inner_diameter_in', 'bamboo_thickness_in', 'area_in2', 'Ix_in4', 'Iy_in4', 'J_in4',
             'Fiber Density']]
    df.rename(
        columns={'outer_diameter_in': '$OD (in)$', 'inner_diameter_in': '$ID (in)$', 'bamboo_thickness_in': '$t (in)$',
                 'area_in2': '$A (in^{2})$', 'Ix_in4': '$I_x (in^4)$', 'Iy_in4': '$I_y (in^4)$', 'J_in4': '$J (in^4)$',
                 'Fiber Density': '$F (\%)$'}, inplace=True)
    if transverse:
        print(df.T.to_latex(float_format="%.2f", escape=False))
    else:
        print(df.to_latex(float_format="%.2f", escape=False))
    if return_df:
        return df


def getBowTieLatex(df_bowtie, culm, internode, transverse=False, return_df=False):
    df = df_bowtie[(df_bowtie.culm == culm) & (df_bowtie.internode <= internode + 2) & (
            df_bowtie.internode >= internode - 2)].reset_index()
    df.loc[pd.isnull(df['max_load (lbf)']), 'max_load (lbf)'] = df.loc[pd.isnull(df['max_load (lbf)']), 'Max load (lb)']
    df = df[['ID', 't_avg (in)', 'l_avg (in)', 'A (in^2)', 'max_load (lbf)', 'Shear Strength (psi)', 'MC (%)']]
    df.sort_values('ID', inplace=True, ignore_index=True)
    df['MC (%)'] = [mc.replace('%', '') for mc in df['MC (%)']]
    df['Shear Strength (psi)'] = df['Shear Strength (psi)'] / 1000
    df.rename(columns={'ID': '\textbf{Sample}', 't_avg (in)': '$t_{avg} (in)$', 'l_avg (in)': '$l_{avg} (in)$',
                       'A (in^2)': '$A (in^2)$', 'max_load (lbf)': '$F (lb)$', 'Shear Strength (psi)': '$\tau_u (ksi)$',
                       'MC (%)': '$MC (\%)$'}, inplace=True)
    if transverse:
        print(df.T.to_latex(float_format="%.2f", escape=False, index=False))
    else:
        print(df.to_latex(float_format="%.2f", escape=False, index=False))
    if return_df:
        return df


def getODLatex(df_diameters, culm, internode, transverse=False, print_info=True, return_df=False):
    df = df_diameters[(df_diameters['ID Culm'] == culm) & (df_diameters['Internode'] == internode)]
    if print_info:
        print('MC:', df['MC (%)'].values[0], '%')
        print('Date:', df['Date'].values[0])
    df = df[['D1 (in) N-S', 'D1 (in) E-W', 'D2 (in) N-S',
             'D2 (in) E-W', 'D3 (in) N-S', 'D3 (in) E-W']]  # .T
    df.index = ['$OD (in)$']
    if transverse:
        df = df.T
        df[['pos', 'dim', 'ori']] = [i.split(' ') for i in df.index]
        df.pos = [pos.replace('D', '') for pos in df.pos]
        df = df.groupby(['pos', 'ori']).mean()
        ls = df.to_latex(float_format="%.2f", escape=False).split('\n')
        del ls[3]
        print('\n'.join(ls))
    else:
        df.columns = [col.replace('D', '').replace(' (in)', '') for col in df.columns]
        print(df.to_latex(float_format="%.2f", escape=False))
    if return_df:
        return df


def read_df_merged(paths, culm, internode):
    for path in paths:
        bn = os.path.basename(path)
        if f'{culm}_{internode}_' in bn[:6]:
            df = pd.read_csv(path)
            df.datetime = pd.to_datetime(df.datetime)
            return df