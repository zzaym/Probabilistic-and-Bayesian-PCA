import numpy             as np
import matplotlib.pyplot as plt
from numpy.random       import multivariate_normal
from ppca               import PPCA

if __name__ == '__main__':
    cov  = np.diag([10] + [1]*31 + [10] + [1]*31)**2
    data = multivariate_normal(np.zeros(64), cov, 1000)
    
    ppca = PPCA(n_dimension=2)
    ppca.fit(data)
    
    plt.matshow(cov);
    plt.title('original covariance matrix')
    plt.show()
    plt.matshow(ppca._C);
    plt.title('fitted covariance matrix')
    plt.show()
    
    plt.scatter(data[:, 0], data[:, 1], alpha=0.2);
    plt.title('original data (first 2 dimensions)')
    plt.show()
    gene = ppca.generate(1000)
    plt.scatter(gene[:, 0], gene[:, 1], alpha=0.2);
    plt.title('generated data (first 2 dimensions)')
    plt.show()