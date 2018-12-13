import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
from ppca                import *
from pca_all_impute      import *

temps_mean = np.loadtxt('mohsst_mean.tsv')
temps_mean = pd.to_numeric(temps_mean.flatten(), errors='coerce').reshape(temps_mean.shape)
monthly_mean = []
for i in range(12):
    monthly_mean.append(temps_mean[i*180:i*180+180, :])
def compress(originalmap):
    compressedmap = np.zeros(shape=(36, 72))
    for i in range(36):
        for j in range(72):
            compressedmap[i, j] = np.nanmean(originalmap[i*5:i*5+5, j*5: j*5+5])
    return compressedmap
for i in range(12):
    monthly_mean[i] = compress(monthly_mean[i])

mohsst = pd.read_csv('mohsst.tsv', delimiter='\t') # this tsv contains the deviations from the monthly mean SST
temp_map_list = []
temps = mohsst.values
def get_order(year, month):
    return (year-1856)*12 + month
def get_index(order):
    start = (order-1) * 73
    return start, start+72
def get_temp_map(year, month, temps, monthly_mean, flat=False):
    ind1, ind2 = get_index(get_order(year, month))
    str2d = temps[ind1:ind2, 1:].T
    num1d = pd.to_numeric(str2d.flatten(), errors='coerce') + monthly_mean[month-1].flatten()
    if flat:
        return num1d
    else:
        return num1d.reshape(36, 72)
for year in range(1856, 1992):
    for month in range(1, 13):
        temp_map_list.append(get_temp_map(year, month, temps, monthly_mean, flat=True))
sst_history = np.array(temp_map_list)


from netCDF4 import Dataset   
dataset = Dataset('sst.mnmean.nc') # this tsv contains the monthly mean SST, but more precise
def get_timeindex_noaasst(year, month):
    return (year-1981)*12 + month-12
def get_latsindex_noaasst(latindex_mohsst):
    start = latindex_mohsst*5
    return start, start+5
def get_lonsindex_noaasst(lonsindex_mohsst):
    start = ( (lonsindex_mohsst+36)%72 ) * 5
    return start, start+5
def get_temp_val(lon_ind, lat_ind, timeindex, var_sst, mean=False):
    lon_lo, lon_hi = get_lonsindex_noaasst(lon_ind)
    lat_lo, lat_hi = get_latsindex_noaasst(lat_ind)
    if mean:
        return np.mean(var_sst[timeindex, lat_lo:lat_hi, lon_lo:lon_hi])
    else:
        return np.array(var_sst[timeindex, lat_lo:lat_hi, lon_lo:lon_hi])
temp_map_list_val = []
for year in range(1982, 1992):
    for month in range(1, 13):
        timeindex = get_timeindex_noaasst(year, month)
        temp_map = np.zeros(shape=(36, 72))
        for lon_ind in range(72):
            for lat_ind in range(36):
                temp_map[lat_ind, lon_ind] = \
                    get_temp_val(lon_ind, lat_ind, timeindex, dataset.variables['sst'], mean=True)
        temp_map_list_val.append(temp_map.flatten())
sst_history_val = np.array(temp_map_list_val) # the validation set for sst

def mse(map1, map2):
    land = np.isnan(map1)
    map2filtered = map2.copy()
    map2filtered[land] = np.nan
    return np.sqrt(np.nansum((map1-map2filtered)**2))/np.sum(~land)
def reestimate(sst_imputed, pcamodel, monthly_mean, month):
    land = np.isnan(monthly_mean[month-1])
    sst_imputed = sst_imputed.reshape(1, -1)
    sst_reest = pcamodel.inverse_transform(pcamodel.transform(sst_imputed)).flatten()
    sst_reest[land.flatten()] = np.nan
    return sst_reest
def calc_mses(sst_history_filled, sst_history_val, pcamodel, monthly_mean): 
    # calculate the mses of reconstructed SSTs
    mses = []
    for year in range(1982, 1992):
        for month in range(1, 13):
            ind = (year-1991)*12+(month-12)-1
            sst_filled = sst_history_filled[ind]
            sst_reest = reestimate(sst_filled, pcamodel, monthly_mean, month)
            sst_val = sst_history_val[ind]
            mses.append(mse(sst_reest, sst_val))
    return mses

import numpy as np
from ppca import PPCA
from sklearn.decomposition import PCA
from bpca import BPCA

class PCAImputer:
    
    def __init__(self, method='pca', n_dimension=0):
        self._q = n_dimension
        self._method = method
        if method == 'pca':
            self._pca = PCA(n_components=self._q)
        elif method == 'ppca':
            self._pca = PPCA(n_dimension=self._q)
        else:
            self._pca = BPCA()
    
    def fit_transform(self, data, ppca_method='eig', probabilistic=False, n_iteration=100, \
                        verbose=False, print_every=10, trace_mse=False, cdata=None):
        self._data     = data.copy() 
        self._missing  = np.isnan(data)
        self._observed = ~self._missing
        self._mse = np.zeros(n_iteration)
              
        row_defau = np.zeros(self._data.shape[0])
        row_means = np.repeat(np.nanmean(self._data, axis=1, out=row_defau).reshape(-1, 1), \
                              self._data.shape[1], axis=1)
        self._data[self._missing] = row_means[self._missing]
        
        
        for i in range(n_iteration):
            if self._method == 'ppca':           
                self._pca.fit(self._data, method=ppca_method)
                self._data[self._missing] = self._pca.inverse_transform(self._pca.transform(self._data, \
                                            probabilistic), probabilistic)[self._missing]
            else:
                self._pca.fit(self._data)
                self._data[self._missing] = self._pca.inverse_transform(self._pca.transform(self._data))[self._missing]
#             self._mse[i] = np.sum((cdata-self._data)**2)/cdata.shape[0]
            if verbose and i % print_every == 0:
                print('Iter %d, MSE=%f' %(i, self._mse[i]))
#             if np.abs(self._mse[i-1]-self._mse[i]) < 1e-3:
#                 break
        return self._data, self._mse

pca_imputer = PCAImputer(method='bpca')
print('DONE!')
sst_history_filled, _ = pca_imputer.fit_transform(sst_history, n_iteration=2, verbose=True)
pcamodel = pca_imputer._pca

import pickle

with open('sst_history_filled.pickle', 'wb') as handle:
    pickle.dump(sst_history_filled, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('pcamodel.pickle', 'wb') as handle:
    pickle.dump(pcamodel, handle, protocol=pickle.HIGHEST_PROTOCOL)