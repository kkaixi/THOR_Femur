# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:42:21 2018
helper functions for thor femur
@author: tangk
"""
import pandas as pd
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
import xlwings as xw
import os
from PMG.read_data import read_table, read_from_common_store
from PMG.COM import arrange


#%% faro 3d plot streams
def check_sheet_names():        
    file_path = 'P:\\Data Analysis\\Data\\driver_knee_faro\\'
    files = os.listdir(file_path)
    sheet_names =  ['RIGHT KNEE CENTERLINE',
                    'IP RIGHT KNEE CENTERLINE',
                    'RIGHT STEERING COLUMN',
                    'LEFT KNEE CENTERLINE',
                    'IP LEFT AT KNEE CENTERLINE',
                    'LEFT KNEE HEIGHT ON IP LOWER',
                    'LEFT KNEE HEIGHT ON IP UPPER']
    missing = []
    for file in files:
        book = xw.Book(file_path + file)
        sheets = [book.sheets[i].name for i in book.sheets]
        for sn in sheet_names:
            if not sn in sheets:
                missing.append([file, sn])
        book.close()
    return missing
            

def retrieve_faro(file_path,file_name,sheet_names):
    book = xw.Book(file_path+file_name)
    data = {}
    for sheet in sheet_names:
        data[sheet] = pd.DataFrame(book.sheets[sheet].range('A1').expand().value,columns=['x','y','z'])
    book.close()
    return data

def draw_faro_stream(tc, title = '', streams=None):
    if streams==None:
        streams =  ['RIGHT KNEE CENTERLINE',
                    'IP RIGHT KNEE CENTERLINE',
                    'RIGHT STEERING COLUMN',
                    'LEFT KNEE CENTERLINE',
                    'IP LEFT AT KNEE CENTERLINE',
                    'LEFT KNEE HEIGHT ON IP LOWER',
                    'LEFT KNEE HEIGHT ON IP UPPER']
    data = retrieve_faro('P:\\Data Analysis\\Data\\driver_knee_faro\\',tc + '.xls',streams)
    traces = []
    for stream in data:
        tr = go.Scatter3d(x = data[stream]['x'].values,
                          y = data[stream]['y'].values,
                          z = data[stream]['z'].values,
                          marker = {'size' : 3,
                                    'color':'#1f77b4'},
                          line = {'width': 2,
                                  'color': '#1f77b4'},
                          name = stream,
                          text = [stream]*len(data[stream]))
        traces.append(tr)
    layout = {'scene': {'aspectratio': {'x':1, 'y': 1, 'z': 1}},
              'title': title,
              'showlegend': False}
    
    fig = {'data': traces, 'layout': layout}
    plot(fig)
    
def sep_faro_axes(data):
    # data is in the format of the output of retrieve_faro
    return pd.DataFrame({k+'_'+ax: data[k][ax] for k in data for ax in data[k]})

def knee_initialize(directory,channels, cutoff, streams=[],tc=None,query=None,filt=None,drop=None):
    # read table
    if 'Table.csv' in os.listdir(directory):
        table = read_table(directory + 'Table.csv')
    else:
        print('No table!')
        return
    if query:
        table = table.query(query)
    if filt:
        table = table.filter(items=filt)
    if tc:
        table = table.loc[tc]
    tc = table.index
    
    # get chdata
    t, fulldata = read_from_common_store(tc, channels)
    chdata = arrange.to_chdata(fulldata,cutoff)
    t = t[cutoff]
    
    # append faro points
    # initialize faro points as would be done for regular data
    if streams==[]:
        streams =  ['RIGHT KNEE CENTERLINE',
                    'IP RIGHT KNEE CENTERLINE',
                    'RIGHT STEERING COLUMN',
                    'LEFT KNEE CENTERLINE',
                    'IP LEFT AT KNEE CENTERLINE',
                    'LEFT KNEE HEIGHT ON IP LOWER',
                    'LEFT KNEE HEIGHT ON IP UPPER']
        
    faro = {}
    for i in tc:
        faro[i] = sep_faro_axes(retrieve_faro('P:\\Data Analysis\\Data\\driver_knee_faro\\',i + '.xls',streams))
    faro = arrange.to_chdata(faro)
    chdata = pd.concat((chdata, faro), axis=1)
    return table, t, chdata

#%% interpolate knee and IP points at knee centerline and estimate shortest distance
#from scipy.spatial.distance import pdist, squareform
#knee_stream = 'LEFT KNEE CENTERLINE'
#ip_stream = 'IP LEFT AT KNEE CENTERLINE'     
#
#for file_name in os.listdir(file_path):
#    data = retrieve_faro(file_path,file_name)
#    
#    x_knee = data[knee_stream]['x'][1:].values
#    y_knee = data[knee_stream]['z'][1:].values
#    
#    dist = np.min(pdist(np.vstack((x_knee,y_knee)).T))
#    chdata.at[file_name.rstrip('.xls'),'femur_dist_left'] = dist