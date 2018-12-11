import numpy             as np
import matplotlib.pyplot as plt
from numpy.random       import multivariate_normal
from ppca               import PPCA

if __name__ == '__main__':
    
    cov  = np.diag([10, 9, 8, 7] + [1]*28 + [6, 5, 4, 3] + [1]*28)**2
    data = multivariate_normal(np.zeros(64), cov, 256)
   
    ppca1 = PPCA(n_dimension=4)
    ppca1.fit(data, method='EM')
    ppca2 = PPCA(n_dimension=4)
    ppca2.fit(data, method='eig')
    
    print('\n\n\n\n**TEST FITTING THE COVARIANCE MATRIX**')
    plt.matshow(cov);
    print('\n\noriginal covariance matrix')
    plt.show()
    plt.matshow(ppca1._C);
    print('\n\nfitted covariance matrix (fitted by EM)')
    plt.show()
    plt.matshow(ppca2._C);
    print('\n\nfitted covariance matrix (fitted by eigen)')
    plt.show()
    
    print('\n\n\n\n**TEST GENERATING DATA**')
    plt.scatter(data[:, 0], data[:, 1], alpha=0.2);
    print('\n\noriginal data (first 2 dimensions)')
    plt.show()
    gene = ppca1.generate(256)
    plt.scatter(gene[:, 0], gene[:, 1], alpha=0.2);
    print('\n\ngenerated data (first 2 dimensions) (fitted by EM)')
    plt.show()
    gene = ppca2.generate(256)
    plt.scatter(gene[:, 0], gene[:, 1], alpha=0.2);
    print('\n\ngenerated data (first 2 dimensions) (fitted by eigen)')
    plt.show()
    
    print('\n\n\n\n**TEST CALCULATING LIKELIHOOD**')
    ppca1 = PPCA(n_dimension=2)
    loglikelihoods = ppca1.fit(data, method='EM', keep_loglikes=True)
    plt.plot(loglikelihoods)
    plt.show()

    print('\n\n\n\n**TEST DIMENSION REDUCTION AND RECOVERING**')
    plt.matshow(data)
    print('\n\noriginal data')
    plt.show()
    
    ppca3 = PPCA(n_dimension=2)
    ppca3.fit(data, method='EM')
    plt.matshow( ppca3.inverse_transform( ppca3.transform(data) ) )
    print('\n\nrecovered data: 2-component')
    plt.show()
    
    ppca4 = PPCA(n_dimension=2)
    ppca4.fit(data, batchsize=16, n_iteration=2000, method='EM')
    plt.matshow( ppca4.inverse_transform( ppca4.transform(data) ) )
    print('\n\nrecovered data: 2-component (mini-batch)')
    plt.show()
    
    ppca5 = PPCA(n_dimension=63)
    ppca5.fit(data, method='EM')
    plt.matshow( ppca5.inverse_transform( ppca5.transform(data) ) )
    print('\n\nrecovered data: 63-component')
    plt.show()
    
    