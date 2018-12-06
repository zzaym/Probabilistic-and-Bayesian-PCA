import numpy             as np
import matplotlib.pyplot as plt
from numpy.random       import multivariate_normal
from ppca               import PPCA

if __name__ == '__main__':
    cov  = np.diag([5, 4, 3, 2, 1, 1, 1, 1, 1, 1])**2
    plt.matshow(cov);
    plt.title('original covariance matrix')
    plt.show()
    data = multivariate_normal(np.zeros(10), cov, 1000)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.2);
    plt.title('original data (first 2 dimension)')
    plt.show()
    
    
    ppca = PPCA(n_dimension=4)
    ppca.fit(data)
    gene = ppca.generate(1000)
    plt.matshow(ppca._C);
    plt.title('fitted covariance matrix')
    plt.show()
    plt.scatter(gene[:, 0], gene[:, 1], alpha=0.2);
    plt.title('generated data (first 2 dimension)')
    plt.show()