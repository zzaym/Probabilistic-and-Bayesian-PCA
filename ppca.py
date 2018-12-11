"""
Probablistic Principal Component Analysis using the EM algorithm from Tipping & Bishop 1997.
(See http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.426.2749&rep=rep1&type=pdf)
"""

import numpy as np
from numpy.random          import randn
from numpy.random          import normal
from numpy.random          import multivariate_normal
from numpy.linalg          import det
from numpy.linalg          import eig
from numpy.linalg          import pinv
from numpy.linalg          import multi_dot
from sklearn.exceptions    import NotFittedError

class PPCA(object):

    def __init__(self, n_dimension):
        
        # mapping from latent variables to observations: x = W z + mu + epsilon
        # where x: observation, x in R^(d)
        # q: dimension of latent space
        self._q = n_dimension 
    
    def fit(self, data, batchsize=None, n_iteration=500, method='EM', 
            keep_loglikes=False):
        
        if method not in ('EM', 'eig'):
            raise ValueError('unrecognized method.')
        if method == 'eig' and keep_loglikes: 
            raise ValueError('loglike not supported for eig method') 
        if method == 'eig' and batchsize is not None:
            raise ValueError('mini-batch not supported for eig method') 
        
        ####################### INITIALIZE OBSERVATIONS #######################
        # X: observations, X in R^(d*N), data assumed to be in R^(N*d)
        self._X = data.T
        # d: dimension of observations
        self._d = self._X.shape[0]
        # N: number of observations
        self._N = self._X.shape[1]
        # mu: mean of x, mu in R^(d)
        self._mu = np.mean(self._X, axis=1).reshape(-1, 1)
        
        ##################### INITIALIZE LATENT VARIABLES #####################     
        # W: linear transformation matrix, W in R^(d*q)
        self._W      = randn(self._d, self._q)
        # epsilon: Gaussian noise, epsilon in R^(d), epsilon ~ N(0, sigma^2 I)
        self._sigma2 = 0
        # C: covariance matrix of observation, x ~ N(mu, C)
        self._update_C()
        
        loglikes = [] if keep_loglikes else None
        
        if   method == 'EM':
            loglikes = self._fit_EM(batchsize, n_iteration, keep_loglikes)
        else: #method == 'eig'
            self._fit_eig(n_iteration) 
        
        return loglikes
    
    def transform(self, data_observ, probabilistic=False):
        invM = pinv(self._calc_M())
        expect_data_latent = multi_dot([invM, self._W.T, 
                                        data_observ.T - self._mu])
        assert expect_data_latent.shape == (self._q, len(data_observ))
        if probabilistic: 
            cov   = np.dot(self._sigma2, invM)
            data_latent = np.zeros(shape=(len(data_observ), self._q))
            for i in range(len(data_observ)):
                data_latent[i] = multivariate_normal(expect_data_latent[:, i].flatten(), cov)
            return data_latent
        else:
            return expect_data_latent.T
    
    def inverse_transform(self, data_latent, probabilistic=False):
        expec_data_observ = np.dot(self._W, data_latent.T) + self._mu
        if probabilistic:
            return (expec_data_observ
                   + normal(scale=np.sqrt(self._sigma2), size=self._X.shape)).T
        else:
            return expec_data_observ.T
        
    def generate(self, n_sample):
        try:
            return multivariate_normal(self._mu.flatten(), self._C, n_sample)
        except:
            raise NotFittedError('This PPCA instance is not fitted yet. Call \'fit\' with appropriate arguments before using this method.')
        
    ######################## FITTING BY EM ALGORITHM ##########################
    def _fit_EM(self, batchsize, n_iteration=500, keep_loglikes=False):
        
        if batchsize is not None and batchsize > self._N:
            raise ValueError('batchsize exceeds number of observations') 
        
        loglikes = [] if keep_loglikes else None
        
        for i in range(n_iteration):
            # E-step: Estimation   (omitted)
            # M-step: Maximization
            if batchsize is not None:
                idx = self.batch_idx(i, batchsize)
                Xb  = self._X[:, idx]
                self._maximize_L(Xb, np.mean(Xb, axis=1).reshape(-1, 1))
            else:
                self._maximize_L(self._X, self._mu)
                
            if keep_loglikes:
                loglikes.append(self._calc_loglike(self._X, self._mu))
        
        return loglikes
    
    def _maximize_L(self, X, mu):
        S = self._calc_S(X, mu)
        M = self._calc_M()
        self._update_W(S, M)
        self._update_sigma2(S, M)
        self._update_C()
    
    def _update_W(self, S, M):
        temp = pinv( self._sigma2 * np.eye(self._q) \
                     + multi_dot([ pinv(M), self._W.T, S, self._W]) )
        self._W = multi_dot([ S, self._W, temp ])
    
    def _update_sigma2(self, S, M):
        temp = multi_dot([ S, self._W, pinv(M), self._W.T ])
        self._sigma2 = 1/self._d * np.trace(S - temp)    
    
    ##################### FITTING BY EIGENDECOMPOSITION #######################
    def _fit_eig(self, n_iteration=500):
        
        S = self._calc_S(self._X, self._mu)
        vals, vecs = eig(S)
        vals, vecs = vals.real, vecs.real
        ordbydom = np.argsort(vals)[::-1]
        topq_dom = ordbydom[:self._q]
        less_dom = ordbydom[self._q:]
        self._sigma2 = np.sum(vals[less_dom]) / (self._d - self._q)
        self._W = np.dot( vecs[:, topq_dom], 
                          np.sqrt(np.diag(vals[topq_dom])-self._sigma2*np.eye(self._q)) )
        self._update_C()
    
    ########################### UTILITY FUNCTIONS #############################
    def _calc_S(self, X, mu):
        centeredX = X - mu
        return np.dot(centeredX, centeredX.T) / X.shape[1]
    
    def _calc_M(self):
        return self._sigma2 * np.eye(self._q) + np.dot(self._W.T, self._W)
    
    def _calc_loglike(self, X, mu):
        return -self._N/2 * (self._d*np.log(2*np.pi) \
               + np.log(det(self._C)) \
               + np.trace(np.dot(pinv(self._C), self._calc_S(X, mu))))
      
    def _update_C(self):
        self._C = self._sigma2 * np.eye(self._d) + np.dot(self._W, self._W.T)
        
    def batch_idx(self, i, batchsize):
        if batchsize == self._N:
            return np.arange(self._N)
        idx1 = (i*batchsize)     % self._N
        idx2 = ((i+1)*batchsize) % self._N
        if idx2 < idx1:    idx1 -= self._N
        return np.arange(idx1, idx2)