import re
import numpy as np
import pandas as pd


def read_man(filename):
    with open(filename, 'r') as f:
        data_man = f.readlines()
        
    for idx, l in enumerate(data_man):
        if l[:2] == '# ':
            data_man[idx] = data_man[idx][2:]
            break
    columns = re.split('\s+', data_man[idx])
    data_man = data_man[idx+1:]
    data_man = pd.DataFrame([re.split('\s+', x) for _, x in enumerate(data_man)], columns=columns).set_index('stno')
    data_man = data_man[['yyyymmddhh','PS01','TX01','RH01','WD01','WD02','PP01']].astype(float)
    data_man['yyyymmddhh'] = (data_man['yyyymmddhh']-1).astype(int).astype(str)
    return data_man
    
def read_auto(filename):
    with open(filename, 'r') as f:
        data_auto = f.readlines()
        
    for idx, l in enumerate(data_auto):
        if l[:2] == '# ':
            data_auto[idx] = data_auto[idx][2:]
            break
    columns = re.split('\s+', data_auto[idx])
    data_auto = data_auto[idx+1:]
    data_auto = pd.DataFrame([re.split('\s+', x) for _,x in enumerate(data_auto)], columns=columns).set_index('stno')
    data_auto = data_auto[['yyyymmddhh','PS01','TX01','RH01','WD01','WD02','PP01']].astype(float)
    data_auto['yyyymmddhh'] = (data_auto['yyyymmddhh']-1).astype(int).astype(str)
    return data_auto

def read_data(data_man, data_auto, sta_list):
    data_man = read_man(filename=data_man)
    data_auto = read_auto(filename=data_auto)
    data = pd.concat([data_man, data_auto], axis=0, sort=False).reset_index()

    data.fillna(-9999, inplace=True)
    data.replace(-9996, np.nan, inplace=True)
    data.fillna(method='bfill', inplace=True)
    data.replace(-9998, 0.1, inplace=True)
    data.replace(-9997, np.nan, inplace=True)
    data.replace(-9999, np.nan, inplace=True)
    data.replace(-9991, np.nan, inplace=True)

    data.replace(-999.8, 0.1, inplace=True)
    data.replace(-999.7, np.nan, inplace=True)
    data.replace(-999.9, np.nan, inplace=True)
    data.replace(-999.1, np.nan, inplace=True)
    data.replace(999.9, np.nan, inplace=True)
    
    data.dropna(inplace=True)
    
    # data.set_index(['yyyymmddhh'], inplace=True)
    data['yyyymmddhh'] = pd.to_datetime(data['yyyymmddhh'],format='%Y%m%d%H')
    data.set_index(['stno','yyyymmddhh'], inplace=True)

    for i in sta_list.index:
        data.loc[i,'lat'] = sta_list.loc[i, 'Latitude']
        data.loc[i,'lon'] = sta_list.loc[i, 'Longitude']
        data.loc[i,'height'] = sta_list.loc[i, 'Height(m)']

    data.reset_index(inplace=True)
    data.set_index(['yyyymmddhh'], inplace=True)
    data.dropna(inplace=True)
    return data

def localP_2_seaP(p: int or list, h, t):
    if type(p) == list:
        p = np.array(list)    
    if type(h) == list:
        h = np.array(list)
    if type(t) == list:
        t = np.array(list)
    return p*10**(h/(18400*(1+t/273)))