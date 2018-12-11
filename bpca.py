# https://pdfs.semanticscholar.org/a1fb/a67f147b16e3c4bffdab3cc6f17520c74547.pdf

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma

class BPCA(object):

    def __init__(self, a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3):
        # hyperparameters
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta
        self.elbos = None
        self.variations = None
        self.loglikelihoods = None


    def update(self):
        self.tau = self.a_tau_tilde / self.b_tau_tilde
        self.alpha = (self.a_alpha_tilde / self.b_alpha_tilde).flatten()
        self.cov_z = np.linalg.inv(np.eye(self.q) + self.tau *
                        (np.trace(self.cov_w) + np.dot(self.mean_w.T, self.mean_w)))
        self.mean_z = self.tau * np.dot(np.dot(self.cov_z, self.mean_w.T), self.Xb - self.mean_mu)
        self.cov_mu = np.eye(self.d) / (self.beta + self.b * self.tau)
        self.mean_mu = self.tau * np.dot(self.cov_mu, np.sum(self.Xb-np.dot(self.mean_w,
                        self.mean_z), axis=1)).reshape(self.d, 1)
        self.cov_w = np.linalg.inv(np.diag(self.alpha) + self.tau *
                        (self.b * self.cov_z + np.dot(self.mean_z, self.mean_z.T)))
        self.mean_w = self.tau * np.dot(self.cov_w, np.dot(self.mean_z, (self.Xb-self.mean_mu).T)).T
        self.b_alpha_tilde = self.b_alpha + (np.trace(self.cov_w) +
                        np.array([np.dot(self.mean_w[:,i], self.mean_w[:,i]) for i in range(self.q)])) / 2
        self.b_tau_tilde = self.b_tau + 0.5 * np.sum(np.array([
                        np.dot(self.Xb[:,i], self.Xb[:,i]) +
                        np.trace(self.cov_mu) + np.dot(self.mean_mu.flatten(), self.mean_mu.flatten()) +
                        np.trace(np.dot(np.trace(self.cov_w)+np.dot(self.mean_w.T, self.mean_w),
                                        self.cov_z+np.outer(self.mean_z[:,i], self.mean_z[:,i]))) +
                        2 * np.dot(np.dot(self.mean_mu.flatten(), self.mean_w), self.mean_z[:,i]) +
                        -2 * np.dot(self.Xb[:,i], np.dot(self.mean_w, self.mean_z[:,i]))+
                        -2 * np.dot(self.Xb[:,i], self.mean_mu.flatten())
                        for i in range(self.b)]))
        
    
    def calculate_log_likelihood(self):
        z = np.array([np.random.multivariate_normal(self.mean_z[:,i], self.cov_z) for i in range(self.b)]).T
        mu = np.random.multivariate_normal(self.mean_mu.flatten(), self.cov_mu)
        w = np.array([np.random.multivariate_normal(self.mean_w[i], self.cov_w) for i in range(self.d)])
        pred = np.dot(w, z) + mu.reshape(-1,1)
        loglikelihood = np.sum(np.array([mvn.logpdf(self.Xb[:,i], pred[:,i], np.eye(self.d)/self.tau) for i in range(self.b)]))
        return loglikelihood


    def calculate_ELBO(self):
        '''ELBO = E_q[-log(q(theta))+log(p(theta)+log(p(Y|theta,X)))]
                = -entropy + logprior + loglikelihood '''

        # random sample
        z = np.array([np.random.multivariate_normal(self.mean_z[:,i], self.cov_z) for i in range(self.b)]).T
        mu = np.random.multivariate_normal(self.mean_mu.flatten(), self.cov_mu)
        w = np.array([np.random.multivariate_normal(self.mean_w[i], self.cov_w) for i in range(self.d)])
        alpha = np.random.gamma(self.a_alpha_tilde, 1/self.b_alpha_tilde)
        tau = np.random.gamma(self.a_tau_tilde, 1/self.b_tau_tilde)

        # entropy
        # q(z)
        entropy = np.sum(np.array([mvn.logpdf(z[:,i], self.mean_z[:,i], self.cov_z) for i in range(self.b)]))

        # q(mu)
        entropy += mvn.logpdf(mu, self.mean_mu.flatten(), self.cov_mu)

        # q(W)
        entropy += np.sum(np.array([mvn.logpdf(w[i], self.mean_w[i], self.cov_w) for i in range(self.d)]))

        # q(alpha)
        entropy += np.sum(gamma.logpdf(alpha, self.a_alpha_tilde, scale=1/self.b_alpha_tilde))

        # q(tau)
        entropy += gamma.logpdf(tau, self.a_tau_tilde, scale=1/self.b_tau_tilde)

        # logprior
        # p(z), z ~ N(0, I)
        logprior = np.sum(np.array([mvn.logpdf(z[:,i], mean=np.zeros(self.q), cov=np.eye(self.q)) for i in range(self.b)]))

        # p(w|alpha), conditional gaussian
        logprior += np.sum(np.array([self.d/2*np.log(alpha[i]/(2*np.pi))-alpha[i]*np.sum(w[:,i]**2)/2 for i in range(self.q)]))

        # p(alpha), alpha[i] ~ Gamma(a, b)
        logprior += np.sum(gamma.logpdf(alpha, self.a_alpha, scale=1/self.b_alpha))

        # p(mu), mu ~ N(0, I/beta)
        logprior += mvn.logpdf(mu, mean=np.zeros(self.d), cov=np.eye(self.d)/self.beta)

        # p(tau), tau ~ Gamma(c, d)
        logprior += gamma.logpdf(tau, self.a_tau, scale=1/self.b_tau)

        # loglikelihood
        pred = np.dot(w, z) + mu.reshape(-1,1)
        loglikelihood = np.sum(np.array([mvn.logpdf(self.Xb[:,i], pred[:,i], np.eye(self.d)/tau) for i in range(self.b)]))

        return -entropy + logprior + loglikelihood


    def batch_idx(self, i):
        if self.b == self.N:
            return np.arange(self.N)
        idx1 = (i*self.b) % self.N
        idx2 = ((i+1)*self.b) % self.N
        if idx2 < idx1:
            idx1 -= self.N
        return np.arange(idx1, idx2)


    def fit(self, X=None, batch_size=128, iters=500, print_every=100, verbose=False, trace_elbo=False, trace_loglikelihood=False):
         # data, # of samples, dims
        self.X = X.T # don't need to transpose X when passing it
        self.d = self.X.shape[0]
        self.N = self.X.shape[1]
        self.q = self.d-1
        self.ed = []
        self.b = min(batch_size, self.N)

        # variational parameters
        self.mean_z = np.random.randn(self.q, self.b) # latent variable
        self.cov_z = np.eye(self.q)
        self.mean_mu = np.random.randn(self.d, 1)
        self.cov_mu = np.eye(self.d)
        self.mean_w = np.random.randn(self.d, self.q)
        self.cov_w = np.eye(self.q)
        self.a_alpha_tilde = self.a_alpha + self.d/2
        self.b_alpha_tilde = np.abs(np.random.randn(self.q, 1))
        self.a_tau_tilde = self.a_tau + self.b * self.d / 2
        self.b_tau_tilde = np.abs(np.random.randn(1))

        # update
        order = np.arange(self.N)
        elbos = np.zeros(iters)
        loglikelihoods = np.zeros(iters)
        for i in range(iters):
            idx = order[self.batch_idx(i)]
            self.Xb = self.X[:,idx]
            self.update()
            if trace_elbo:
                elbos[i] = self.calculate_ELBO()
            if trace_loglikelihood:
                loglikelihoods[i] = self.calculate_log_likelihood()
            if verbose and i % print_every == 0:
                print('Iter %d, ELBO: %f, alpha: %s' % (i, elbos[i], str(self.alpha)))
        self.captured_dims()
        self.elbos = elbos if trace_elbo else None
        self.loglikelihoods = loglikelihoods if trace_loglikelihood else None


    def captured_dims(self):
        sum_alpha = np.sum(1/self.alpha)
        self.ed = np.array([i for i, inv_alpha in enumerate(1/self.alpha) if inv_alpha > sum_alpha/self.q])


    def transform(self, X=None):
        X = self.X if X is None else X.T
        w = self.mean_w[:, self.ed]
        m = np.eye(len(self.ed))/self.tau + np.dot(w.T, w)
        inv_m = np.linalg.inv(m)
        z = np.dot(np.dot(inv_m, w.T), X - self.mean_mu)
        return np.array([np.random.multivariate_normal(z[:,i], inv_m/self.tau) for i in range(X.shape[1])])


    def inv_transform(self, n):
        w = self.mean_w[:, self.ed]
        c = np.eye(self.d)/self.tau + np.dot(w, w.T)
        return np.array([np.random.multivariate_normal(self.mean_mu.flatten(), c) for i in range(n)])


    def fit_transform(self, X=None, batch_size=128, iters=500, print_every=100, verbose=False, trace_elbo=False, trace_loglikelihood=False):
        self.fit(X, batch_size, iters, print_every, verbose, trace_elbo)
        return self.transform()


    def get_weight_matrix(self):
        return self.mean_w


    def get_inv_variance(self):
        return self.alpha


    def get_effective_dims(self):
        return len(self.ed)


    def get_elbo(self):
        return self.elbos


    def get_loglikelihood(self):
        return self.loglikelihoods

