import numpy             as np
import matplotlib.pyplot as plt
from numpy.random        import multivariate_normal
from pca_impute          import PCAImputer

if __name__ == '__main__':
    
    # original data
    cov  = np.diag([10] + [1]*31 + [10] + [1]*31)**2
    data = multivariate_normal(np.zeros(64), cov, 64)
    
    # missing at random
    mask_missing = np.random.randint(2, size=data.shape)
    data_missing = data.copy()
    data_missing[np.where(mask_missing)] = np.nan
    
    # impute by PCA 
    imputer = PCAImputer(n_dimension=2)
    data_imputed = imputer.fit_transform(data_missing)
    
    plt.matshow(data)
    plt.title('original data')
    plt.show()
    plt.matshow(data_missing)
    plt.title('missing at random')
    plt.show()
    plt.matshow(data_imputed)
    plt.title('imputed data')
    plt.show()