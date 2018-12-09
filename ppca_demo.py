import numpy             as np
import matplotlib.pyplot as plt
from numpy.random       import multivariate_normal
from ppca               import PPCA

if __name__ == '__main__':
    cov  = np.diag([10] + [1]*31 + [10] + [1]*31)**2
    data = multivariate_normal(np.zeros(64), cov, 64)
   
    ppca1 = PPCA(n_dimension=2)
    ppca1.fit(data, method='EM')
    ppca2 = PPCA(n_dimension=2)
    ppca2.fit(data, method='eig')
    
    plt.matshow(cov);
    plt.title('original covariance matrix')
    plt.show()
    plt.matshow(ppca1._C);
    plt.title('fitted covariance matrix (fitted by EM algorithm)')
    plt.show()
    plt.matshow(ppca2._C);
    plt.title('fitted covariance matrix (fitted by eigendecomposition)')
    plt.show()
    
    plt.scatter(data[:, 0], data[:, 1], alpha=0.2);
    plt.title('original data (first 2 dimensions)')
    plt.show()
    gene = ppca1.generate(1000)
    plt.scatter(gene[:, 0], gene[:, 1], alpha=0.2);
    plt.title('generated data (first 2 dimensions) (fitted by EM algorithm)')
    plt.show()
    gene = ppca2.generate(1000)
    plt.scatter(gene[:, 0], gene[:, 1], alpha=0.2);
    plt.title('generated data (first 2 dimensions) (fitted by eigendecomposition)')
    plt.show()
    
    ppca1 = PPCA(n_dimension=2)
    loglikelihoods = ppca1.fit(data, method='EM', keep_likelihoods=True)
    plt.plot(loglikelihoods[2:])
    plt.show()
 
    plt.matshow(data)
    plt.title('original data')
    plt.show()
    
    ppca3 = PPCA(n_dimension=2)
    ppca3.fit(data, method='EM')
    plt.matshow( ppca3.inverse_transform( ppca3.transform(data) ) )
    plt.title('recovered data: 2-component')
    plt.show()
    
    ppca4 = PPCA(n_dimension=63)
    ppca4.fit(data, method='EM')
    plt.matshow( ppca4.inverse_transform( ppca4.transform(data) ) )
    plt.title('recovered data: 63-component')
    plt.show()