import numpy               as np
from sklearn.decomposition import PCA # should be changed to PPCA later

class PCAImputer:
    
    def __init__(self, n_dimension):
        self._q = n_dimension
    
    def fit_transform(self, data, n_iteration=100):
        self._data     = data.copy() 
        self._missing  = np.isnan(data)
        self._observed = ~self._missing
        self._pca      = PCA(n_components = self._q)
        
        row_defau = np.zeros(self._data.shape[0])
        row_means = np.repeat(np.nanmean(self._data, axis=1, out=row_defau).reshape(-1, 1), \
                              self._data.shape[1], axis=1)
        self._data[self._missing] = row_means[self._missing]
        for i in range(n_iteration):
            self._pca.fit(self._data)
            self._data[self._missing] = self._pca.inverse_transform( \
                                        self._pca.transform(self._data) )[self._missing]
        return self._data
    