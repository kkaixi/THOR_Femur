# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:44:28 2018

@author: tangk
"""
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from string import ascii_lowercase, ascii_uppercase
from THOR_Femur_helper import *
from PMG.COM.helper import *
from PMG.COM.get_props import *
from PMG.COM.plotfuns import *
from plotly.offline import plot
import plotly.graph_objs as go

directory = 'P:\\Data Analysis\\Projects\\THOR Femur\\'
streams =  ('LEFT KNEE CENTERLINE',
            'IP LEFT AT KNEE CENTERLINE',
            'LEFT KNEE HEIGHT ON IP UPPER',
            'LEFT KNEE HEIGHT ON IP LOWER')
knee_cols = ['LEFT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]
ip_cols = ['IP LEFT AT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]
channels = ['11FEMRLE00THFOZB',
            '10CVEHCG0000ACXD',
            '11LUSP0000THFOXA',
            '11LUSP0000THFOYA',
            '11LUSP0000THFOZA',
            '11LUSP0000THMOXA',
            '11LUSP0000THMOYA',
            '11ILACLE00THFOXA',
            '11PELV0000THACXA',
            '11PELV0000THACYA',
            '11PELV0000THACZA',
            '11PELV0000THAVXA',
            '11PELV0000THAVYA',
            '11PELV0000THAVZA',
            '11ACTBLE00THFOYB',
            '11ACTBLE00THFOZB',
            '11FEMRLE00THFOXB',
            '11FEMRLE00THFOYB',
            '11FEMRLE00THMOXB',
            '11FEMRLE00THMOYB',
            '11FEMRLE00THMOZB',
            '11KNSLLE00THDSXC',
            '11TIBILEUPTHFOXB',
            '11TIBILEUPTHFOYB',
            '11TIBILEUPTHFOZB',
            '11TIBILEUPTHFORB',
            '11TIBILEUPTHMOXB',
            '11TIBILEUPTHMOYB',
            '11TIBILEUPTHMORB',
            '11TIBILEMITHACXA',
            '11TIBILEMITHACYA',
            '11TIBILELOTHFOXB',
            '11TIBILELOTHFOYB',
            '11TIBILELOTHFOZB',
            '11TIBILELOTHMOXB',
            '11TIBILELOTHMOYB',
            '11ANKLLE00THANXA',
            '11ANKLLE00THANYA',
            '11ANKLLE00THANZA',
            '11FOOTLE00THACXA',
            '11FOOTLE00THACYA',
            '11FOOTLE00THACZA',
            '11SEBE0000B6FO0D']
cutoff = range(100, 1000)
#%%
table, t, chdata = knee_initialize(directory,channels, cutoff, streams=streams)
table_full = pd.read_csv(directory + 'Table.csv',index_col=0)
i_to_t = get_i_to_t(t)
#%% preprocessing: find dash angle (angle from the vertical where +tive is CW), find key points, and get distances from key points
def sep_coords(data,dim1 = 'x', dim2 = 'z'):
    if isinstance(data,np.ndarray):
        return data[0], data[1]
    elif isinstance(data,pd.core.series.Series):
        stream = data.index[0][:-1]
        return data[stream + dim1], data[stream + dim2]


def get_max_angle_anchored(data):
    # data should be an array-like of [x_coords, y_coords]
    # returns the maximum angle between the first point and the outermost
    # point in the stream
    x, y = sep_coords(data)
    i = np.argmax(x)
    return np.degrees(np.arctan(np.abs(x[i]-x[0])/np.abs(y[i]-y[0])))


def get_end2end_angle(data):
    # data should be an array-like of [x_coords, y_coords]
    # data specifies the coordinates recorded on the IP
    # returns the angle between the first and last points of the stream
    # in data
    x, y = sep_coords(data)
    return np.degrees(np.arctan(np.abs(x[-1]-x[0])/np.abs(y[-1]-y[0])))

def get_angle_from_top(data):
    """largest angle from the top of the IP. Angle is measured CCW from 6 o'clock"""
    x, y = sep_coords(data)
    if np.argmax(x)==len(x)-1:
        i = np.argmin(x)
    else:
        i = np.argmax(x)
        
    dx = x[i] - x[-1]
    dz = np.abs(y[-1]-y[i])
    return np.degrees(np.arctan(dx/dz))

def key_points_by_distance(data, ref=0, dist=0):
    # data is a pd.Series of points on the knee
    # returns the coordinates of the intersection between
    # the outline of the knee and the horizontal line drawn dist points
    # above ref 
    # ref is the vertical coordinate of the reference point
    x, y = sep_coords(data)

    x_horz = np.linspace(1500,2000)
    y_horz = np.array([ref+dist]*len(x_horz))
    
    xkey, ykey = intersection(x,y,x_horz,y_horz)
    if len(xkey)==0:
        return np.nan, np.nan
    else:
        return xkey[0], ykey[0]

def ip_key_point_from_angle(data,ref,angle):
    # data is a pd.Series of IP points
    # ref is a tuple of reference point
    # angle is the angle (duh)
    # returns the coordinates of the intersection between the points in data
    # and the line drawn from ref towards the IP at an angle angle
    
    x, y = sep_coords(data)
    
    delta_x = 200
    delta_y = delta_x*np.tan(np.radians(angle))
    
    ref_x = np.linspace(ref[0],ref[0]-delta_x)
    ref_y = np.linspace(ref[1],ref[1]+delta_y)

    xkey, ykey = intersection(x,y,ref_x,ref_y)
    if len(xkey)==0:
        return np.nan, np.nan
    else:
        return xkey[0], ykey[0]

def ip_points_from_centre(data,ref=0,dist=0):
    x, y = sep_coords(data,dim2='y')
    
    x_horz = np.linspace(1000,2000)
    y_horz = np.array([ref+dist]*len(x_horz))
    
    xkey, ykey = intersection(x,y,x_horz,y_horz)
    if len(xkey)==0:
        return np.nan, np.nan
    else:
        return xkey[0], ykey[0]
    
def get_ip_angle(chdata,method='max'):
    """gets angle of ip from vertical using method specified"""
    if method=='max':
        chdata['left_ip_angle'] = chdata[['IP LEFT AT KNEE CENTERLINE_x','IP LEFT AT KNEE CENTERLINE_z']].apply(get_max_angle_anchored,axis=1)
    elif method=='end2end':
        chdata['left_ip_angle'] = chdata[['IP LEFT AT KNEE CENTERLINE_x','IP LEFT AT KNEE CENTERLINE_z']].apply(get_end2end_angle,axis=1)
    else:
        print('method not valid!')
        return
    return chdata
    

def draw_ip_lines(chdata, ip_deltas):
    """draws fake ip lines and adds it to chdata
    returns chdata with added columns and legend specifying what the letters mean"""
    ind_iter = iter(ascii_uppercase)
    legend = {}
    
    for i in ip_deltas:
        ip_ind = next(ind_iter)
        legend[ip_ind] = 'on_ip_' + str(i) + 'mm_from_center'
        colnames = [ip_ind + coord for coord in ['_x','_y','_z']]
        chdata = chdata.assign(**{col: np.nan for col in colnames})
        chdata[colnames] = chdata[colnames].astype(object)
        
        for tc in chdata.index:
            p = pd.DataFrame(columns=['x','y','z'], index=['p1','p2'])
            
            p.loc['p1',['x','y']] = ip_points_from_centre(chdata.loc[tc,['LEFT KNEE HEIGHT ON IP UPPER_x','LEFT KNEE HEIGHT ON IP UPPER_y']],
                                       ref=np.mean(chdata.at[tc,'IP LEFT AT KNEE CENTERLINE_y']),
                                       dist=i) # point on top line
            p.at['p1','z'] = np.mean(chdata.at[tc,'LEFT KNEE HEIGHT ON IP UPPER_z'])
            
            p.loc['p2',['x','y']] = ip_points_from_centre(chdata.loc[tc,['LEFT KNEE HEIGHT ON IP LOWER_x','LEFT KNEE HEIGHT ON IP LOWER_y']],
                                       ref=np.mean(chdata.at[tc,'IP LEFT AT KNEE CENTERLINE_y']),
                                       dist=i) # point on bottom line
            p.at['p2','z'] = np.mean(chdata.at[tc,'LEFT KNEE HEIGHT ON IP LOWER_z'])
            
            tmp = pd.DataFrame(columns=colnames)
            for col in colnames:
                tmp[col] = np.linspace(p.at['p1',col[-1]],p.at['p2',col[-1]])
            chdata.loc[tc,colnames] = tmp.apply(tuple).apply(np.array)
    return chdata, legend

def get_knee_kps(chdata, knee_kp_deltas):
    """gets key points on knee and adds it to chdata
    returns chdata with added columns and legend specifying what the letters mean"""
    ind_iter = iter(ascii_lowercase)
    legend = {}
  
    #find the farthest inward point
    ind = next(ind_iter)
    for axis in ['_x','_y','_z']:
        chdata[ind+axis] = chdata['LEFT KNEE CENTERLINE'+axis].apply(lambda x: x[1])
        legend[ind] = 'kp_farthest_backward'
        
    # find key points using distance from the farthest inward point
    for dist in knee_kp_deltas:
        ind = next(ind_iter)
        for tc in chdata.index: 
            ref = chdata.at[tc,'LEFT KNEE CENTERLINE_z'][1]
            
            xkey, ykey = key_points_by_distance(chdata.loc[tc,knee_cols],ref,dist)
            chdata.at[tc,ind+'_x'] = xkey
            chdata.at[tc,ind+'_z'] = ykey
            legend[ind] = 'kp_dist_'+str(dist)
    return chdata, legend


def get_distances(chdata, ip_streams, knee_kps, angles):
    """gets distances between key points on knee and ip at angles specified
    returns chdata with distance info
    ip_streams is the legend output of draw_ip_lines
    knee_kps is the legend output of get_knee_kps
    angles is the range of angles to try"""
    ip_streams = list(ip_streams)
    knee_kps = list(knee_kps)
    if 'IP LEFT AT KNEE CENTERLINE' not in ip_streams:
        ip_streams.append('IP LEFT AT KNEE CENTERLINE')
    for tc in chdata.index:
        for kp in knee_kps:
            for ip in ip_streams: 
                ip_cols = [ip + coord for coord in ['_x','_y','_z']]
                for angle in angles:
                    ref_x = chdata.at[tc,kp+'_x']
                    ref_y = chdata.at[tc,kp+'_z']
                    xkey, ykey = ip_key_point_from_angle(chdata.loc[tc,ip_cols],(ref_x,ref_y),angle)
                    dist = np.sqrt((ref_x-xkey)**2+(ref_y-ykey)**2)
                    chdata.at[tc, kp + ip + str(angle) + 'deg_x'] = xkey
                    chdata.at[tc, kp + ip + str(angle) + 'deg_z'] = ykey
                    chdata.at[tc, kp + ip + str(angle) + 'deg_dist'] = dist    
    return chdata


def get_knee_angle(data):
    """gets the angle of the leg from the first two points on the knee.
    The angle is CCW from the horizontal (3'oclock)"""
    x, z = sep_coords(data)
    dx = abs(x[1]-x[0])
    dz = abs(z[1]-z[0])
    return np.degrees(np.arctan(dz/dx))
#%% set parameters. 
ip_deltas = range(-60, 30, 10)    
knee_kp_deltas = range(10,70,10)
angles = range(1)

chdata, ip_legend = draw_ip_lines(chdata, ip_deltas)
chdata, kp_legend = get_knee_kps(chdata, knee_kp_deltas)
chdata = get_distances(chdata, ip_legend, kp_legend, angles)
chdata['left_knee_angle'] = chdata[['LEFT KNEE CENTERLINE_' + i for i in ['x','z']]].apply(get_knee_angle,axis=1)
chdata['angle_from_top'] = chdata[['IP LEFT AT KNEE CENTERLINE_' + i for i in ['x','z']]].apply(get_angle_from_top,axis=1)

#%% write points to csv

t2peak_femur = chdata['11FEMRLE00THFOZB'].apply(get_argmin).dropna().astype(int)
for tc in t2peak_femur.index:
    cols = chdata.columns[~chdata.loc[tc].apply(lambda x: isinstance(x, np.float64)).values]
    chdata.loc[tc, cols] = chdata.loc[tc, cols].apply(lambda x: x[:t2peak_femur[tc]])

feature_funs = {'Min_': [get_min],
                'Max_': [get_max]}
features = pd.concat(chdata[channels].chdata.get_features(feature_funs).values(),axis=1,sort=True)

dist_features = chdata[[i for i in chdata.columns if '_dist' in i]]
dist_features['left_ip_angle_end2end'] = get_ip_angle(chdata,'end2end')['left_ip_angle']
dist_features['left_ip_angle_max'] = get_ip_angle(chdata,'max')['left_ip_angle']
dist_features['min_distance_from_a'] = chdata[[i for i in chdata.columns if 'a' in i and 'dist' in i]].min(axis=1)
dist_features['min_distance_from_b'] = chdata[[i for i in chdata.columns if 'b' in i and 'dist' in i]].min(axis=1)
dist_features['min_distance_from_c'] = chdata[[i for i in chdata.columns if 'c' in i and 'dist' in i]].min(axis=1)
dist_features['min_distance_from_d'] = chdata[[i for i in chdata.columns if 'd' in i and 'dist' in i]].min(axis=1)
dist_features['min_distance_from_e'] = chdata[[i for i in chdata.columns if 'e' in i and 'dist' in i]].min(axis=1)
dist_features['min_distance_from_f'] = chdata[[i for i in chdata.columns if 'f' in i and 'dist' in i]].min(axis=1)
dist_features['min_distance_from_g'] = chdata[[i for i in chdata.columns if 'g' in i and 'dist' in i]].min(axis=1)
dist_features['delta_veh_cg'] = features.loc[table.index, 'Min_10CVEHCG0000ACXD'] - features.loc[table['PAIR'].values,'Min_10CVEHCG0000ACXD'].values
dist_features['ratio_veh_cg'] = features.loc[table.index, 'Min_10CVEHCG0000ACXD']/features.loc[table['PAIR'].values,'Min_10CVEHCG0000ACXD'].values
dist_features = dist_features.loc[:, (dist_features.count()>(len(features)//2))]
features = pd.concat((features, dist_features), axis=1)
#features.to_csv(directory + 'features.csv')
#%% plot femur loads vs. various distances and compute r2
#angle_method = 'max' # one of 'max', 'end2end'
#distance_method = 'min' # one of 'min', 'max', 'orig' 
#x_list = [features['min_distance_from_b'],
#          features['ratio_veh_cg']]
#groups = {'grp1': table.index}
#
#for factor in x_list:
#    x = {}
#    y = {}
#    for grp in groups:
#        y[grp] = features.loc[groups[grp], 'left_femur_load']
#        x[grp] = factor.loc[groups[grp]]
#
#    
##    if len(x['grp'].dropna())<=10:
##        continue
##    
#    spearmanr = rho(x['grp1'], y['grp1'])
#    pearsonr = corr(x['grp1'], y['grp1'])
#    rsq = r2(x['grp1'], y['grp1'])
##    if np.isnan(spearmanr) or np.isnan(pearsonr) or np.isnan(rsq):
##        break
##    if rsq<0.2:
##        continue
##    if max(spearmanr,pearsonr) < 0.4 and min(spearmanr,pearsonr)>-0.4:
##        continue
#    
#    print(factor.name)
#    print('rho=' + str(spearmanr) + ', R=' + str(pearsonr) + ', R2=' + str(rsq))
#    fig = plot_scatter_with_labels(x, y)
#    fig = set_labels_plotly(fig, {'xlabel': factor.name, 'ylabel': 'Femur Load', 'legend': {}})
#    plot(fig)

#%%
#import plotly.graph_objs as go
#trace = go.Scatter3d(x=features['min_distance_from_b'],
#                     y=features['ratio_veh_cg'],
#                     z=features['left_femur_load'],
#                     text=table.index,
#                     mode='markers')
#data = [trace]
#plot(data)

#%% initiate JSON file
#to_JSON = {'project_name': 'THOR_Femur',
#           'directory': directory}
#
#with open(directory+'params.json','w') as json_file:
#    json.dump(json_file)
#%% plot
#chdata_norm = chdata[['LEFT KNEE CENTERLINE_x','LEFT KNEE CENTERLINE_z', 
#                      'IP LEFT AT KNEE CENTERLINE_x', 'IP LEFT AT KNEE CENTERLINE_z']]
#chdata_norm[[i for i in chdata_norm.columns if '_x' in i]] = chdata_norm[[i for i in chdata_norm.columns if '_x' in i]].sub(chdata['a_x'],'index')
#chdata_norm[[i for i in chdata_norm.columns if '_z' in i]] = chdata_norm[[i for i in chdata_norm.columns if '_z' in i]].sub(chdata['a_z'],'index')
##chdata_norm[[i for i in chdata_norm.columns if '_x' in i]] = chdata_norm[[i for i in chdata_norm.columns if '_x' in i]].apply(lambda x: x-x['IP LEFT AT KNEE CENTERLINE_x'][0],axis=1)
##chdata_norm[[i for i in chdata_norm.columns if '_z' in i]] = chdata_norm[[i for i in chdata_norm.columns if '_z' in i]].apply(lambda x: x-x['IP LEFT AT KNEE CENTERLINE_z'][0],axis=1)
#
#subset = table.query('KAB==\'NO\' and DUMMY==\'THOR\'')
#cmap = matplotlib.cm.get_cmap('cool')
#normalize = matplotlib.colors.Normalize(vmin=features['left_femur_load_plus_x1'].min(),vmax=features['left_femur_load_plus_x1'].max())
#
#fig, ax = plt.subplots(figsize=(12,10))
#for tc in subset.index:
#    ax.plot(chdata_norm.at[tc,'LEFT KNEE CENTERLINE_x'],
#             chdata_norm.at[tc,'LEFT KNEE CENTERLINE_z'],
#             color=cmap(normalize(features.at[tc,'left_femur_load_plus_x1'])))
#    ax.plot(chdata_norm.at[tc,'IP LEFT AT KNEE CENTERLINE_x'],
#             chdata_norm.at[tc,'IP LEFT AT KNEE CENTERLINE_z'],
#             color=cmap(normalize(features.at[tc,'left_femur_load_plus_x1'])),
#             label=subset.at[tc,'MODEL'])
#cax, _ = matplotlib.colorbar.make_axes(ax)
#cbar = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=normalize)
#ax.legend()
#%%
#for tc in chdata.index:
#    draw_faro_stream(tc,title=tc)

#%%
#import lightgbm as lgb
#lgb_drop = ['Max_11ILACLE00THFOXA',
#            'Min_10CVEHCG0000ACXD']
##
##lgb_drop2 = []
##
##lgb_drop = lgb_drop + lgb_drop2
#
#
#
#x = features.loc[table.drop('TC18-212').query('DUMMY==\'THOR\' and KAB==\'NO\' and SPEED==48').index].drop(lgb_drop, axis=1)
#y = x.pop('Min_11FEMRLE00THFOZB')
#
## dimensionality reduction of x
#x = x.loc[:,(x.count()>(len(x)//2))] # remove features with a lot of missing values
#corr = x.corr().abs()>0.6 # remove columns with high correlation
#drop_cols = []
#nfeat = len(corr.columns)
#for i, col in enumerate(corr.columns):
#    drop = corr[col].values[i+1:]
#    colnames = corr.columns.values[i+1:]
#    drop = colnames[drop]
#    # figure out which one to drop based on which gives the highest R2 score
#    for d in drop:
#        if r2(x[col], y) > r2(x[d], y):
#            drop_cols.append(d)
#        else:
#            drop_cols.append(col)
##    drop_cols.append(colnames[drop])
##drop_cols = np.unique(np.concatenate(drop_cols))
#x = x.drop(list(dict.fromkeys(drop_cols)), axis=1)
#
#train_data = lgb.Dataset(x, label=y)
#
#param = {'objective': 'regression',
#         'feature_fraction': 0.8,
#         'min_data': 1,
#         'min_data_in_bin': 1}
#
#n_rounds = 10
#importance = pd.DataFrame(index=range(n_rounds), columns=x.columns)
#
#for i in range(n_rounds):
#    model = lgb.train(param, train_data)
#    importance.loc[i] = model.feature_importance()
##    lgb.plot_importance(model, figsize=(10,8))
#print(importance.mean().sort_values())

#%% iteratively add regressors
from sklearn.linear_model import LinearRegression
rthresh = 1
rsq = 0
drop = ['Min_11SEBE0000B6FO0D']
indices = table.drop('TC18-212').query('DUMMY==\'THOR\' and KAB==\'YES\' and SPEED==48').index
y = features.loc[indices, 'Min_11FEMRLE00THFOZB']
best_features = []
feature_list = features.columns.drop(y.name)
feature_list = feature_list.drop(best_features)
corr = features.corr().abs()>0.5


while rsq<rthresh and len(best_features)<=10:
    lr = LinearRegression()
    rsq_list = pd.Series(index=feature_list)
    for col in feature_list:
        if features.loc[indices, col].count() <= len(y)//2:
            feature_list = feature_list.drop(col)
            continue
        x = features.loc[indices, [col] + best_features]
        i = ~(x.isna().any(axis=1)).values.flatten()
        lr = lr.fit(x[i], y[i])
        rsq_list[col] = lr.score(x[i], y[i])
    if len(rsq_list)==0 or rsq_list.isna().all(): continue
    feature_add = rsq_list.idxmax()
    rsq = rsq_list[feature_add]
    best_features.append(feature_add)
    
    feature_list = feature_list.drop(feature_add)
    feature_list = feature_list.drop([i for i in corr[feature_add][corr[feature_add]].index if i in feature_list])# also remove features with high correlations to the feature added
    print('adding feature {0}. updated rsq {1}'.format(feature_add, rsq))
print(best_features)

#%%
trace = go.Scatter3d(x=features.loc[indices,'Max_11TIBILEMITHACYA'],
                     y=features.loc[indices,'Max_11ACTBLE00THFOYB'],
                     z=features.loc[indices,'Min_11FEMRLE00THFOZB'],
                     mode='markers',
                     text=indices)
data = [trace]
layout = {'scene': {'xaxis': {'title': 'Tibia'},
                    'yaxis': {'title': 'Actb'},
                    'zaxis': {'title': 'Femur'}}}

fig = go.Figure(data=data, layout=layout)
plot(fig)