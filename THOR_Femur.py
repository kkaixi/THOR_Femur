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
from sklearn.linear_model import Lars, LarsCV, LassoLars, LassoLarsCV 
from sklearn.preprocessing import StandardScaler
import re

directory = 'P:\\Data Analysis\\Projects\\THOR Femur\\'
streams =  ('LEFT KNEE CENTERLINE',
            'IP LEFT AT KNEE CENTERLINE',
            'LEFT KNEE HEIGHT ON IP UPPER',
            'LEFT KNEE HEIGHT ON IP LOWER',
            'RIGHT KNEE CENTERLINE',
            'IP RIGHT KNEE CENTERLINE',
            'RIGHT STEERING COLUMN')
knee_cols = ['LEFT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]
ip_cols = ['IP LEFT AT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]
channels = ['10CVEHCG0000ACXD','11SEBE0000B6FO0D',
            '11LUSP0000THFOXA','11LUSP0000THFOYA','11LUSP0000THFOZA',
            '11LUSP0000THMOXA','11LUSP0000THMOYA',
            '11ILACLE00THFOXA','11ILACRI00THFOXA'
            '11PELV0000THACXA','11PELV0000THACYA','11PELV0000THACZA',
            '11PELV0000THAVXA','11PELV0000THAVYA','11PELV0000THAVZA',
            '11ACTBLE00THFOYB','11ACTBLE00THFOZB','11ACTBRI00THFOYB','11ACTBRI00THFOZB',
            '11FEMRLE00THFOXB','11FEMRLE00THFOYB','11FEMRLE00THFOZB',
            '11FEMRRI00THFOXB','11FEMRRI00THFOYB','11FEMRRI00THFOZB',
            '11FEMRLE00THMOXB','11FEMRLE00THMOYB','11FEMRLE00THMOZB',
            '11FEMRRI00THMOXB','11FEMRRI00THMOYB','11FEMRRI00THMOZB',
            '11KNSLLE00THDSXC','11KNSLRI00THDSXC',
            '11TIBILEUPTHFOXB','11TIBILEUPTHFOYB','11TIBILEUPTHFOZB','11TIBILEUPTHFORB',
            '11TIBIRIUPTHFOXB','11TIBIRIUPTHFOYB','11TIBIRIUPTHFOZB','11TIBIRIUPTHFORB',
            '11TIBILEUPTHMOXB','11TIBILEUPTHMOYB','11TIBILEUPTHMORB',
            '11TIBIRIUPTHMOXB','11TIBIRIUPTHMOYB','11TIBIRIUPTHMORB',
            '11TIBILEMITHACXA','11TIBILEMITHACYA','11TIBIRIMITHACXA','11TIBIRIMITHACYA',
            '11TIBILELOTHFOXB','11TIBILELOTHFOYB','11TIBILELOTHFOZB',
            '11TIBIRILOTHFOXB','11TIBIRILOTHFOYB','11TIBIRILOTHFOZB',
            '11TIBILELOTHMOXB','11TIBILELOTHMOYB','11TIBIRILOTHMOXB','11TIBIRILOTHMOYB',
            '11ANKLLE00THANXA','11ANKLLE00THANYA','11ANKLLE00THANZA',
            '11ANKLRI00THANXA','11ANKLRI00THANYA','11ANKLRI00THANZA',
            '11FOOTLE00THACXA','11FOOTLE00THACYA','11FOOTLE00THACZA',
            '11FOOTRI00THACXA','11FOOTRI00THACYA','11FOOTRI00THACZA']
cutoff = range(100, 1000)
#%%
table, t, chdata = knee_initialize(directory,channels, cutoff, streams=streams)
table_full = pd.read_csv(directory + 'Table.csv',index_col=0)
i_to_t = get_i_to_t(t)

# preprocessing
chdata.loc['TC18-216', ['IP LEFT AT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]] = chdata.loc['TC18-216', ['IP LEFT AT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]].apply(np.flip)
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
    notna = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x, y = x[notna], y[notna]
    i = np.argmax(x)
    return np.degrees(np.arctan(np.abs(x[i]-x[0])/np.abs(y[i]-y[0])))


def get_end2end_angle(data):
    # data should be an array-like of [x_coords, y_coords]
    # data specifies the coordinates recorded on the IP
    # returns the angle between the first and last points of the stream
    # in data
    x, y = sep_coords(data)
    notna = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x, y = x[notna], y[notna]
    return np.degrees(np.arctan(np.abs(x[-1]-x[0])/np.abs(y[-1]-y[0])))

def get_angle_from_top(data):
    """largest angle from the top of the IP. Angle is measured CCW from 6 o'clock"""
    x, y = sep_coords(data)
    notna = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x, y = x[notna], y[notna]
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

def ip_key_point_from_angle(data,ref,angle,coords=['x','z'],sgn=-1):
    # data is a pd.Series of IP points
    # ref is a tuple of reference point
    # angle is the angle (duh)
    # returns the coordinates of the intersection between the points in data
    # and the line drawn from ref towards the IP at an angle angle
    # coords are the axes that the angles are calculated from
    # the angle is 0 degrees along coords[0] and 90 degrees along coords[1]
    # sgn is the sign of delta_x (to control the direction of 0 degrees)
    
    x, y = sep_coords(data, dim1=coords[0], dim2=coords[1])
    
    delta_x = 200
    delta_y = delta_x*np.tan(np.radians(angle))
    
    ref_x = np.linspace(ref[0],ref[0]+sgn*delta_x)
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
    
def get_ip_angle(cols,method='max'):
    """gets angle of ip from vertical using method specified
    cols is a dataframe of the x and z coordinates"""
    if method=='max':
        angles = cols.apply(get_max_angle_anchored,axis=1)
    elif method=='end2end':
        angles = cols.apply(get_end2end_angle,axis=1)
    else:
        print('method not valid!')
        return
    return angles
    

def draw_ip_lines(upper, lower, center_ref, ip_deltas):
    """draws fake ip lines and adds it to chdata
    returns dataframe with artificial streams and legend specifying what the letters mean
    upper and lower are the respective streams on the IP
    center_ref is the IP at knee centerline"""
    ind_iter = iter(ascii_uppercase)
    legend = {}
    ip_streams = {}
    
    for i in ip_deltas:
        ip_ind = next(ind_iter)
        legend[ip_ind] = 'on_ip_' + str(i) + 'mm_from_center'
        colnames = [ip_ind + coord for coord in ['_x','_y','_z']]        
        ip_streams.update({col: pd.Series(index=center_ref.index).astype('object') for col in colnames})
        
        for tc in center_ref.index:
            p = pd.DataFrame(columns=['x','y','z'], index=['p1','p2'])
            
            p.loc['p1',['x','y']] = ip_points_from_centre(upper.loc[tc, [i for i in upper.columns if '_x' in i or '_y' in i]],
                                       ref=np.nanmean(center_ref[tc]),
                                       dist=i) # point on top line
            p.at['p1','z'] = np.nanmean(upper.filter(regex='_z', axis=1).squeeze()[tc])
            
            p.loc['p2',['x','y']] = ip_points_from_centre(lower.loc[tc, [i for i in lower.columns if '_x' in i or '_y' in i]],
                                       ref=np.nanmean(center_ref[tc]),
                                       dist=i) # point on bottom line
            p.at['p2','z'] = np.nanmean(lower.filter(regex='_z', axis=1).squeeze()[tc])
            for col in colnames:
                ip_streams[col][tc] = tuple(np.linspace(p.at['p1', col[-1]], p.at['p2', col[-1]]))
    ip_streams = pd.DataFrame(ip_streams).applymap(np.array)
    return ip_streams, legend

def get_knee_kps(cols, knee_kp_deltas, ind_iter=None):
    """returns key points on knee with a legend specifying what the letters mean
    cols is the columsn with the coordinates of the knee
    ind_iter is the optional iterator to use for the legend"""
    if ind_iter is None:
        ind_iter = iter(ascii_lowercase)
    legend = {}
    kps = {}
    
    #find the farthest inward point
    ind = next(ind_iter)
    for k in cols.columns:
        kps[ind + k[-2:]] = cols[k].apply(lambda x: x[1])
    legend[ind] = 'kp_farthest_backward'
        
    # find key points using distance from the farthest inward point
    for dist in knee_kp_deltas:
        ind = next(ind_iter)
        kps[ind + '_x'] = pd.Series(index=cols.index)
        kps[ind + '_y'] = pd.Series(index=cols.index)
        kps[ind + '_z'] = pd.Series(index=cols.index)
        for tc in cols.index: 
            ref = cols.filter(regex='_z', axis=1).squeeze()[tc][1]
            
            xkey, ykey = key_points_by_distance(cols.loc[tc],ref,dist)
            kps[ind + '_x'][tc] = xkey
            kps[ind + '_y'][tc] = np.nanmean(cols.filter(regex='_y', axis=1).squeeze()[tc])
            kps[ind + '_z'][tc] = ykey
        legend[ind] = 'kp_dist_'+str(dist)
    kps = pd.DataFrame(kps)
    return kps, legend


def get_distances(kps, streams, angles, coords=['x','z'], sgn=-1):
    """gets distances between key points on knee and ip streams at angles specified
    returns chdata with distance info
    angles is the range of angles to try
    coords are the axes along which the angle calculation is made, with 
    sgn*coords[0] corresponding to 0 degrees"""
    knee_kps = kps.filter(regex='_x', axis=1).columns.map(lambda x: x.rstrip('_x'))
    ip_streams = streams.filter(regex='_x', axis=1).columns.map(lambda x: x.rstrip('_x'))
    distances = {kp + ip + str(angle) + dist: pd.Series(index=kps.index) for kp in knee_kps for ip in ip_streams for angle in angles for dist in ['deg_x', 'deg_y', 'deg_z', 'deg_dist']}
    
    for tc in kps.index:
        for kp in knee_kps:
            for ip in ip_streams:
                ip_cols = [ip + coord for coord in ['_x','_y','_z']]
                if streams.loc[tc, ip_cols].apply(is_all_nan).any(): continue
                for angle in angles:
                    ref_x = kps.at[tc, '_'.join((kp, coords[0]))]
                    ref_y = kps.at[tc, '_'.join((kp, coords[1]))]
                    const_ax = [i for i in ['x','y','z'] if i not in coords][0]
                    ref_z = kps.at[tc, '_'.join((kp, const_ax))]
                    xkey, ykey = ip_key_point_from_angle(streams.loc[tc, ip_cols], (ref_x, ref_y), angle, coords=coords, sgn=sgn)
                    zkey = np.nanmean(streams.loc[tc, '_'.join((ip, const_ax))])
                    dist = np.sqrt((ref_x-xkey)**2+(ref_y-ykey)**2+(ref_z-zkey)**2)
                    distances[kp + ip + str(angle) + 'deg_' + coords[0]][tc] = xkey
                    distances[kp + ip + str(angle) + 'deg_' + coords[1]][tc] = ykey
                    distances[kp + ip + str(angle) + 'deg_' + const_ax][tc] = zkey
                    distances[kp + ip + str(angle) + 'deg_dist'][tc] = dist
    distances = pd.DataFrame(distances)
    return distances


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
angles = range(0, 20, 5)
features = []
#%% get distances for left femur
stream, ip_legend = draw_ip_lines(chdata[['LEFT KNEE HEIGHT ON IP UPPER_' + ax for ax in ['x','y','z']]], 
                                  chdata[['LEFT KNEE HEIGHT ON IP LOWER_' + ax for ax in ['x','y','z']]], 
                                  chdata['IP LEFT AT KNEE CENTERLINE_y'], ip_deltas)
chdata = pd.concat((chdata, stream), axis=1)

kp_coords, kp_legend = get_knee_kps(chdata[knee_cols], knee_kp_deltas)
features.append(kp_coords.rename(lambda x: 'LE_' + x, axis=1))

distances = get_distances(kp_coords, pd.concat((stream, chdata[['IP LEFT AT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]]), axis=1), angles)
features.append(distances.rename(lambda x: 'LE_' + x, axis=1))

for kp in kp_legend:
    features.append(distances[[i for i in distances.columns if kp in i and 'dist' in i]].min(axis=1).rename('LE_min_distance_from_{0}'.format(kp)))
    features.append(distances[[i for i in distances.columns if kp in i and 'dist' in i]].max(axis=1).rename('LE_max_distance_from_{0}'.format(kp)))
    # get min, max, mean, median distance holding deg constant
    for angle in angles:
        features.append(distances[[i for i in distances.columns if kp in i and str(angle) + 'deg_dist' in i]].min(axis=1).rename('LE_min_distance_from_{0}_at_{1}deg'.format(kp, angle)))
        features.append(distances[[i for i in distances.columns if kp in i and str(angle) + 'deg_dist' in i]].max(axis=1).rename('LE_max_distance_from_{0}_at_{1}deg'.format(kp, angle)))
    # get min, max, mean, median distance holding IP constant
    for ip in list(ip_legend) + ['IP LEFT AT KNEE CENTERLINE']:
        features.append(distances[[i for i in distances.columns if kp in i and 'dist' in i and ip in i]].min(axis=1).rename('LE_min_distance_from_{0}_to_{1}'.format(kp, ip)))
        features.append(distances[[i for i in distances.columns if kp in i and 'dist' in i and ip in i]].max(axis=1).rename('LE_max_distance_from_{0}_to_{1}'.format(kp, ip)))
for ip in list(ip_legend) + ['IP LEFT AT KNEE CENTERLINE']:
    features.append(distances[[i for i in distances.columns if ip in i and 'dist' in i]].min(axis=1).rename('LE_min_distance_to_{0}'.format(ip)))
    features.append(distances[[i for i in distances.columns if ip in i and 'dist' in i]].max(axis=1).rename('LE_max_distance_to_{0}'.format(ip)))
#%% get distances for right femur
kp_coords, kp_legend = get_knee_kps(chdata[[i.replace('LEFT', 'RIGHT') for i in knee_cols]], knee_kp_deltas)
features.append(kp_coords.rename(lambda x: 'RI_' + x, axis=1))

distances = get_distances(kp_coords, chdata[['IP RIGHT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]], angles)
features.append(distances.rename(lambda x: 'RI_' + x, axis=1))

sw_distances = get_distances(kp_coords, chdata[['RIGHT STEERING COLUMN_' + ax for ax in ['x','y','z']]], angles, coords=['y','z'], sgn=-1)
features.append(sw_distances.rename(lambda x: 'RI_' + x, axis=1))

for kp in kp_legend:
    features.append(distances[[i for i in distances.columns if kp in i and 'dist' in i]].min(axis=1).rename('RI_min_distance_from_{0}'.format(kp)))
    features.append(distances[[i for i in distances.columns if kp in i and 'dist' in i]].max(axis=1).rename('RI_max_distance_from_{0}'.format(kp)))
    # get min, max, mean, median distance holding deg constant
    for angle in angles:
        features.append(distances[[i for i in distances.columns if kp in i and str(angle) + 'deg_dist' in i]].min(axis=1).rename('RI_min_distance_from_{0}_at_{1}deg'.format(kp, angle)))
        features.append(distances[[i for i in distances.columns if kp in i and str(angle) + 'deg_dist' in i]].max(axis=1).rename('RI_max_distance_from_{0}_at_{1}deg'.format(kp, angle)))
    # get min, max, mean, median distance holding IP constant
    for ip in list(ip_legend) + ['IP RIGHT KNEE CENTERLINE']:
        features.append(distances[[i for i in distances.columns if kp in i and 'dist' in i and ip in i]].min(axis=1).rename('RI_min_distance_from_{0}_to_{1}'.format(kp, ip)))
        features.append(distances[[i for i in distances.columns if kp in i and 'dist' in i and ip in i]].max(axis=1).rename('RI_max_distance_from_{0}_to_{1}'.format(kp, ip)))
for ip in list(ip_legend) + ['IP RIGHT KNEE CENTERLINE']:
    features.append(distances[[i for i in distances.columns if ip in i and 'dist' in i]].min(axis=1).rename('RI_min_distance_to_{0}'.format(ip)))
    features.append(distances[[i for i in distances.columns if ip in i and 'dist' in i]].max(axis=1).rename('RI_max_distance_to_{0}'.format(ip)))
features.append(sw_distances[[i for i in sw_distances.columns if 'dist' in i]].min(axis=1).rename('RI_min_distance_to_sw'))
features.append(sw_distances[[i for i in sw_distances.columns if 'dist' in i]].max(axis=1).rename('RI_max_distance_to_sw'))
#%% write points to csv

# re cut off so that each tc is cut off to the time of peak femur load
left_t2peak = chdata['11FEMRLE00THFOZB'].apply(get_argmin).dropna().astype(int)
right_t2peak = chdata['11FEMRRI00THFOZB'].apply(get_argmin).dropna().astype(int)
cutoff = pd.concat((left_t2peak, right_t2peak), axis=1).max(axis=1)
for tc in cutoff.index:
    cols = chdata.columns[~chdata.loc[tc].apply(lambda x: isinstance(x, np.float64)).values]
    chdata.loc[tc, cols] = chdata.loc[tc, cols].apply(lambda x: x[:cutoff[tc]])

feature_funs = {'Min_': [get_min],
                'Max_': [get_max]}
response_features = pd.concat(chdata[channels].chdata.get_features(feature_funs).values(),axis=1,sort=True)
features.append(response_features)
features.append(get_ip_angle(chdata[['IP LEFT AT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]], 'end2end').rename('LE_ip_angle_end2end'))
features.append(get_ip_angle(chdata[['IP LEFT AT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]], 'max').rename('LE_ip_angle_max'))
features.append(get_ip_angle(chdata[['IP RIGHT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]], 'end2end').rename('RI_ip_angle_end2end'))
features.append(get_ip_angle(chdata[['IP RIGHT KNEE CENTERLINE_' + ax for ax in ['x','y','z']]], 'max').rename('RI_ip_angle_max'))
features.append((response_features.loc[table.index, 'Min_10CVEHCG0000ACXD'] - response_features.loc[table['PAIR'].values,'Min_10CVEHCG0000ACXD'].values).rename('delta_veh_cg'))
features.append((response_features.loc[table.index, 'Min_10CVEHCG0000ACXD']/response_features.loc[table['PAIR'].values,'Min_10CVEHCG0000ACXD'].values).rename('ratio_veh_cg'))
features.append(1/(response_features.loc[table.index, 'Min_10CVEHCG0000ACXD'] - response_features.loc[table['PAIR'].values,'Min_10CVEHCG0000ACXD'].values).rename('1/delta_veh_cg'))
features.append(1/(response_features.loc[table.index, 'Min_10CVEHCG0000ACXD']/response_features.loc[table['PAIR'].values,'Min_10CVEHCG0000ACXD'].values).rename('1/ratio_veh_cg'))

features = pd.concat(features, axis=1)

features = features.loc[:, (features.count())>len(features)//2] # get rid of features with too many na's 
#features.to_csv(directory + 'features.csv')

#%% initiate JSON file
#to_JSON = {'project_name': 'THOR_Femur',
#           'directory': directory}
#
#with open(directory+'params.json','w') as json_file:
#    json.dump(json_file)
#%% Lasso LARS 
KAB = 'YES'
femr = 'LE'


indices = table.drop('TC18-212').query('DUMMY==\'THOR\' and SPEED==48 and KAB==\'{0}\''.format(KAB)).index
r = re.compile('^{0}.*[^xyz]$'.format(femr))
cols = [i for i in features.columns if r.search(i) or 'veh' in i.lower()]
drop = [i for i in cols if i[4:6]=='11'] + ['Max_10CVEHCG0000ACXD']
y = features.loc[indices, 'Min_11FEMR{0}00THFOZB'.format(femr)]
x = features.loc[indices, cols]
if y.name in x.columns:
    x = x.drop(y.name, axis=1)
x = x.drop([i for i in drop if i in x.columns], axis=1)
for col in x:
    x[col] = x[col].replace(np.nan, x[col].mean())
    
ss = StandardScaler()
x = pd.DataFrame(ss.fit_transform(x), columns=x.columns)
corr = x.corr().abs()

for i in range(1, 30):
#    model = LassoLarsCV(max_iter=i, cv=3)
    model = LassoLars(alpha=20, max_iter=i)
    model = model.fit(x, y)
    score = model.score(x, y)
    if score==0:
        print('Model with max_iter={0} gives score of 0. Exiting...'.format(i))
        break
    coefs = pd.Series(model.coef_, index=x.columns)
    keep_cols = coefs[coefs.abs()>0]
    if len(keep_cols.index)==len(x.columns):
        print('No features left to fit. Exiting with an R2 score of {0}'.format(score))
        break
    drop_cols = corr[keep_cols.index]
    drop_cols = drop_cols.drop([i for i in keep_cols.index if i in drop_cols.index])
    drop_cols = drop_cols[drop_cols>0.4].dropna(how='all').index
    drop_cols = [i for i in drop_cols if i in x.columns]
    x = x.drop(drop_cols, axis=1)
    if x.shape[1]==0:
        print('No features left to fit. Exiting with an R2 score of {0}'.format(score))
        break
    print('iteration {0}. Selecting {1} columns. Dropping {2} columns'.format(i, len(keep_cols.index), len(drop_cols)))
    
print('Selecting columns {0} with a CV R2 score of {1}.'.format(keep_cols.index, score))
print(keep_cols)

#%% stepwise regression
def stepwise_regression(x, y, rthresh=0.8, feature_thresh=4, corr_thresh=0.5, drop=[], include_features=[]):
    best_features = []
    best_features.extend(include_features)
    rsq = 0
    feature_list = x.columns.drop([i for i in drop if i in x.columns])
    feature_list = feature_list.drop([i for i in best_features if i in feature_list])
    corr = x.corr().abs()>corr_thresh
    
    while rsq<rthresh and len(best_features)<=feature_thresh and len(feature_list)>0:
        lr = LinearRegression()
        rsq_list = pd.Series(index=feature_list)
        for col in feature_list:
            if x[col].count() <= 4:
                feature_list = feature_list.drop(col)
                continue
            xin = x[[col] + best_features]
            i = np.logical_and(~(xin.isna().any(axis=1)).values.flatten(), ~y.isna().values)
            lr = lr.fit(xin[i], y[i])
            rsq_list[col] = lr.score(xin[i], y[i])
        if len(rsq_list)==0 or rsq_list.isna().all(): break
        feature_add = rsq_list.idxmax()
        rsq = rsq_list[feature_add]
        best_features.append(feature_add)
        feature_list = feature_list.drop(feature_add)
        feature_list = feature_list.drop([i for i in corr[feature_add][corr[feature_add]].index if i in feature_list])
        print('adding feature {0}. updated rsq {1}'.format(feature_add, rsq))
    return best_features, rsq
    
    
pct_selection = 0.9
niter = 10
indices = table.drop('TC18-212').query('DUMMY==\'THOR\' and KAB==\'NO\' and SPEED==48').index
y = features.loc[indices, 'Left_aC5deg_dist']
x = features.loc[indices].drop(y.name, axis=1)
drop = [i for i in features.columns if i[4]!='1']
drop.extend(['Max_10CVEHCG0000ACXD'])

best_features = []
subsets = []
i = 0
while i <= niter:
    print(i)
    sel = np.random.choice(indices, int(len(indices)*pct_selection), replace=False).tolist()
    if sel in subsets:
        continue
    best_features.append(stepwise_regression(x.loc[sel], y.loc[sel], drop=drop))
    subsets.append(sel)
    i = i + 1

#%%
trace = go.Scatter3d(x=features.loc[indices,'Right_max_distance_to_sw'],
                     y=features.loc[indices,'Min_10CVEHCG0000ACXD'],
                     z=features.loc[indices,'Min_11FEMRRI00THFOZB'],
                     mode='markers',
                     text=indices)
data = [trace]
layout = {'scene': {'xaxis': {'title': 'distance'},
                    'yaxis': {'title': 'veh cg'},
                    'zaxis': {'title': 'Femur'}}}

fig = go.Figure(data=data, layout=layout)
plot(fig)