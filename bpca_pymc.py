import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from theano.tensor.nlinalg import diag


class BPCA(object):

    def __init__(self, X, a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3):
        # data, # of samples, dims
        self.X = X
        self.d = self.X.shape[1]
        self.N = self.X.shape[0]
        self.q = self.d-1

        # hyperparameters
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta

        with pm.Model() as model:
            z = pm.MvNormal('z', mu=np.zeros(self.q), cov=np.eye(self.q), shape=(self.N, self.q))
            mu = pm.MvNormal('mu', mu=np.zeros(self.d), cov=np.eye(self.d)/self.beta, shape=self.d)
            alpha = pm.Gamma('alpha', alpha=self.a_alpha, beta=self.b_alpha, shape=self.q)
            w = pm.MatrixNormal('w', mu=np.zeros((self.d, self.q)), rowcov=np.eye(self.d), colcov=diag(1/alpha), shape=(self.d, self.q))
            tau = pm.Gamma('tau', alpha=self.a_tau, beta=self.b_tau)
            x = pm.math.dot(z, w.T) + mu
            obs_x = pm.MatrixNormal('obs_x', mu=x, rowcov=np.eye(self.N), colcov=np.eye(self.d)/tau, shape=(self.N, self.d), observed=self.X)

        self.model = model

        
    def fit(self, iters=10000):
        with self.model:
            inference = pm.ADVI()
            approx = pm.fit(n=iters, method=inference)
            trace = approx.sample(iters//2)

        # save
        s = len(trace)//2
        self.trace = trace
        self.inference = inference
        self.z = trace[s::]['z'].mean(axis=0)
        self.mu = trace[s::]['mu'].mean(axis=0)
        self.alpha = trace[s::]['alpha'].mean(axis=0)
        self.w = trace[s::]['w'].mean(axis=0)


    def transform(self):
    	x = pm.sample_ppc(self.trace, 5000, model=self.model)
    	return x['obs_x'].mean(axis=0)


    def fit_transform(self, iters=10000):
    	self.fit(iters)
    	return self.transform()


    def get_weight_matrix(self):
    	return self.w


    def get_inv_variance(self):
    	return self.alpha


    def get_elbos(self):
    	return -self.inference.hist