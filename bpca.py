# https://pdfs.semanticscholar.org/a1fb/a67f147b16e3c4bffdab3cc6f17520c74547.pdf

import numpy as np
import matplotlib.pyplot as plt


class BPCA(object):

    def __init__(self, X, a_alpha=1e-3, b_alpha=1e-3, \
                 a_tau=1e-3, b_tau=1e-3, beta=1e-3):
        # data, # of samples, dims
        self.X = X.reshape(-1,1) if len(X.shape) < 2 else X
        self.d = self.X.shape[0]
        self.N = self.X.shape[1]

        # hyperparameters
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta

        # variational parameters
        self.mean_z = np.random.randn(self.d, self.N) # latent variable
        self.cov_z = np.random.randn(self.d, self.d)
        self.mean_mu = np.random.randn(self.d, 1)
        self.cov_mu = np.random.randn(self.d, self.d)
        self.mean_w = np.random.randn(self.d, self.d)
        self.cov_w = np.random.randn(self.d, self.d)
        self.a_alpha_tilde = self.a_alpha + self.d/2
        self.b_alpha_tilde = np.abs(np.random.randn(self.d, 1))
        self.a_tau_tilde = self.a_tau + self.N * self.d / 2
        self.b_tau_tilde = np.abs(np.random.randn(1))

    def update(self):
        self.tau = self.a_tau_tilde / self.b_tau_tilde
        self.alpha = (self.a_alpha_tilde / self.b_alpha_tilde).flatten()
        self.cov_z = np.linalg.inv(np.eye(self.d) + self.tau *
                        (np.trace(self.cov_w) + np.dot(self.mean_w.T, self.mean_w)))
        self.mean_z = self.tau * np.dot(np.dot(self.cov_z, self.mean_w.T), self.X - self.mean_mu)
        self.cov_mu = np.eye(self.d) / (self.beta + self.N * self.tau)
        self.mean_mu = self.tau * np.dot(self.cov_mu, np.sum(self.X-np.dot(self.mean_w,
                        self.mean_z), axis=1)).reshape(self.d, 1)
        self.cov_w = np.linalg.inv(np.diag(self.alpha) + self.tau *
                        (self.N * self.cov_z + np.dot(self.mean_z, self.mean_z.T)))
        self.mean_w = self.tau * np.dot(self.cov_w, np.dot(self.mean_z, (self.X-self.mean_mu).T))
        self.b_alpha_tilde = self.b_alpha + (np.trace(self.cov_w) +
                        np.array([np.dot(self.mean_w[:,i], self.mean_w[:,i]) for i in range(self.d)])) / 2
        self.b_tau_tilde = self.b_tau + 0.5 * np.sum(np.array([
                        np.dot(self.X[:,i], self.X[:,i]) +
                        np.trace(self.cov_mu) + np.dot(self.mean_mu.flatten(), self.mean_mu.flatten()) +
                        np.trace(np.dot(np.trace(self.cov_w)+np.dot(self.mean_w.T, self.mean_w),
                                        self.cov_z+np.outer(self.mean_z[:,i], self.mean_z[:,i]))) +
                        2 * np.dot(np.dot(self.mean_mu.flatten(), self.mean_w), self.mean_z[:,i]) +
                        -2 * np.dot(self.X[:,i], np.dot(self.mean_w, self.mean_z[:,i]))+
                        -2 * np.dot(self.X[:,i], self.mean_mu.flatten())
                        for i in range(self.N)]))

    def fit(self, iters=500):
        for i in range(iters):
            self.update()
            if i % 100 == 0:
                print('Iter %d, alpha: %s' % (i+1, str(self.alpha)))

    def hinton(self, matrix=None, max_weight=None, ax=None):
        """Draw Hinton diagram for visualizing a weight matrix."""
        if matrix is None:
            matrix = self.mean_w
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


cov = np.diag(np.array([5,4,3,2,1,1,1,1,1,1])**2)
data = np.random.multivariate_normal(np.zeros(10), cov, size=100)
bpca = BPCA(data.T)
bpca.fit()
bpca.hinton()