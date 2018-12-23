# BPCA

### Overview
This repository is for the final project of Harvard AM205 "Numerical Methods". We examined two generalized versions of conventional PCA from a statistical perspective: Probabilistic PCA (PPCA) and Bayesian PCA (BPCA). We compared their behaviors on synthetic data and real-world data with dif- ferent distributions, and also explored the possible application for estimating missing data.

### Code Specification
#### Python files:
<ul>
    <li> <i>bpca.py</i>: an implementation of Bayesian PCA by varational inference</li>
    <li> <i>bpca_pymc.py</i>: another implementation of Bayesian PCA using pymc</li>
    <li> <i>vis_utils.py</i>: visulization utilities (Hinton graph)</li>
    <li> <i>ppca.py</i>: an implementation of probabilistic PCA</li>   
    <li> <i>ppca_demo.py</i>: a demo for ppca.py</li>
    <li> <i>pca_impute.py</i>: a imputer built on top of ppca.py</li>
    <li> <i>pca_impute_demo.py</i>: a demo for pca_impute.py</li>
    <li> <i>pca_all_impute.py</i>: another imputer that integrated PCA PPCA and BPCA</li>
</ul>

#### Jupyter notebooks (for test and demo):

<ul>
   <li> <i>General Test.ipynb</i>: code for experiments section in the final report</li>
   <li> <i>plots.ipynb</i>: code for experiments section in the final report</li>
   <li> <i>BPCA Test.ipynb</i>: experiments on the behavior of Bayesian PCA</li>
   <li> <i>Imputation Test</i>.ipynb: experiments on the behavior of PCAImputer</li>
   <li> <i>recovering_SST.ipynb</i>: code for recovering sea suface temperature section in the final report, the data for this notebook can be found at https://drive.google.com/drive/folders/1gaVxsZJZfinTenCBicCRggJygEA50D7S?usp=sharing</li>
</ul>
