import numpy             as np
import matplotlib.pyplot as plt
from numpy.random        import multivariate_normal
from pca_impute          import PCAImputer

if __name__ == '__main__':
    
    # original data
    cov  = np.diag([10, 9, 8, 7] + [1]*28 + [6, 5, 4, 3] + [1]*28)**2
    data = multivariate_normal(np.zeros(64), cov, 256)
    
    # missing at random
    mask_missing = np.random.randint(2, size=data.shape)
    data_missing = data.copy()
    data_missing[np.where(mask_missing)] = np.nan
    
    # impute by PCA 
    imputer = PCAImputer(n_dimension=8)
    data_imputed = imputer.fit_transform(data_missing, n_iteration=100)
    
    plt.matshow(data)
    plt.title('original data')
    plt.show()
    plt.matshow(data_missing)
    plt.title('missing at random')
    plt.show()
    plt.matshow(data_imputed)
    plt.title('imputed data')
    plt.show()
    
    print('reconstruction err: {}'.format(np.sqrt(np.sum(np.sum(np.square(data-data_imputed))))))