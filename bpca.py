# https://pdfs.semanticscholar.org/a1fb/a67f147b16e3c4bffdab3cc6f17520c74547.pdf

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma

class BPCA(object):

    def __init__(self, X, a_alpha=1e-3, b_alpha=1e-3, \
                 a_tau=1e-3, b_tau=1e-3, beta=1e-3, var_noise=1):
        # data, # of samples, dims
        self.X = X.reshape(-1,1) if len(X.shape) < 2 else X
        self.d = self.X.shape[0]
        self.N = self.X.shape[1]
        self.q = self.d-1

        # hyperparameters
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta
        self.var_noise = var_noise

        # variational parameters
        self.mean_z = np.random.randn(self.q, self.N) # latent variable
        self.cov_z = np.eye(self.q)
        self.mean_mu = np.random.randn(self.d, 1)
        self.cov_mu = np.eye(self.d)
        self.mean_w = np.random.randn(self.d, self.q)
        self.cov_w = np.eye(self.q)
        self.a_alpha_tilde = self.a_alpha + self.d/2
        self.b_alpha_tilde = np.abs(np.random.randn(self.q, 1))
        self.a_tau_tilde = self.a_tau + self.N * self.d / 2
        self.b_tau_tilde = np.abs(np.random.randn(1))


    def update(self):
        self.tau = self.a_tau_tilde / self.b_tau_tilde
        self.alpha = (self.a_alpha_tilde / self.b_alpha_tilde).flatten()
        self.cov_z = np.linalg.inv(np.eye(self.q) + self.tau *
                        (np.trace(self.cov_w) + np.dot(self.mean_w.T, self.mean_w)))
        self.mean_z = self.tau * np.dot(np.dot(self.cov_z, self.mean_w.T), self.X - self.mean_mu)
        self.cov_mu = np.eye(self.d) / (self.beta + self.N * self.tau)
        self.mean_mu = self.tau * np.dot(self.cov_mu, np.sum(self.X-np.dot(self.mean_w,
                        self.mean_z), axis=1)).reshape(self.d, 1)
        self.cov_w = np.linalg.inv(np.diag(self.alpha) + self.tau *
                        (self.N * self.cov_z + np.dot(self.mean_z, self.mean_z.T)))
        self.mean_w = self.tau * np.dot(self.cov_w, np.dot(self.mean_z, (self.X-self.mean_mu).T)).T
        self.b_alpha_tilde = self.b_alpha + (np.trace(self.cov_w) +
                        np.array([np.dot(self.mean_w[:,i], self.mean_w[:,i]) for i in range(self.q)])) / 2
        self.b_tau_tilde = self.b_tau + 0.5 * np.sum(np.array([
                        np.dot(self.X[:,i], self.X[:,i]) +
                        np.trace(self.cov_mu) + np.dot(self.mean_mu.flatten(), self.mean_mu.flatten()) +
                        np.trace(np.dot(np.trace(self.cov_w)+np.dot(self.mean_w.T, self.mean_w),
                                        self.cov_z+np.outer(self.mean_z[:,i], self.mean_z[:,i]))) +
                        2 * np.dot(np.dot(self.mean_mu.flatten(), self.mean_w), self.mean_z[:,i]) +
                        -2 * np.dot(self.X[:,i], np.dot(self.mean_w, self.mean_z[:,i]))+
                        -2 * np.dot(self.X[:,i], self.mean_mu.flatten())
                        for i in range(self.N)]))
        

    def calculate_ELBO(self):
        '''ELBO = E_q[-log(q(theta))+log(p(theta)+log(p(Y|theta,X)))]
                = -entropy + logprior + loglikelihood '''

        # random sample
        z = np.array([np.random.multivariate_normal(self.mean_z[:,i], self.cov_z) for i in range(self.N)]).T
        mu = np.random.multivariate_normal(self.mean_mu.flatten(), self.cov_mu)
        w = np.array([np.random.multivariate_normal(self.mean_w[i], self.cov_w) for i in range(self.d)])
        alpha = np.random.gamma(self.a_alpha_tilde, 1/self.b_alpha_tilde)
        tau = np.random.gamma(self.a_tau_tilde, 1/self.b_tau_tilde)

        # entropy
        # q(z)
        entropy = np.sum(np.array([mvn.logpdf(z[:,i], self.mean_z[:,i], self.cov_z) for i in range(self.N)]))

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
        logprior = np.sum(np.array([mvn.logpdf(z[:,i], mean=np.zeros(self.q), cov=np.eye(self.q)) for i in range(self.N)]))

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
        loglikelihood = np.sum(np.array([mvn.logpdf(self.X[:,i], pred[:,i], self.var_noise*np.eye(self.d)) for i in range(self.N)]))

        return -entropy + logprior + loglikelihood


    def calculate_varation(self):
        z = np.array([np.random.multivariate_normal(self.mean_z[:,i], self.cov_z) for i in range(self.N)]).T
        mu = np.random.multivariate_normal(self.mean_mu.flatten(), self.cov_mu)
        w = np.array([np.random.multivariate_normal(self.mean_w[i], self.cov_w) for i in range(self.d)])
        pred = np.dot(w, z) + mu.reshape(-1,1)
        mse = np.sum((self.X-pred)**2)
        return mse / np.sum(self.X**2)


    def fit(self, iters=500, print_every=100, trace_elbo=False, trace_varation=False):
        elbos = np.zeros(iters)
        variations = np.zeros(iters)
        for i in range(iters):
            self.update()
            if trace_elbo:
                elbos[i] = self.calculate_ELBO()
            if trace_varation:
                variations[i] = self.calculate_varation()
            if i % print_every == 0:
                print('Iter %d, ELBO: %f, alpha: %s' % (i, elbos[i], str(self.alpha)))
        return [elbos if trace_elbo else None, variations if trace_varation else None]


    def hinton(self, matrix=None, max_weight=None, ax=None):
        """Draw Hinton diagram for visualizing a weight matrix."""
        if matrix is None:
            matrix = self.mean_w.T
            # matrix = np.zeros_like(self.mean_w)
            # for i in range(self.d):
            #     matrix[:,i] = np.random.multivariate_normal(self.mean_w[:,i], self.cov_w)
        ax = ax if ax is not None else plt.gca()

        if not max_weight:
            max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()
        plt.show()