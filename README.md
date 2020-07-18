# An Accelerated Method for Derivative-Free Smooth Stochastic Convex Optimization
## [arXiv](https://arxiv.org/abs/1802.09022)
Implementation of the algorithms and experiments from the paper ***"An Accelerated Method for Derivative-Free Smooth Stochastic Convex Optimization"*** by Eduard Gorbunov, Pavel Dvurechensky, Alexander Gasnikov.


## Files
* algorithms.py contains the implementations of the methods considered in the experimental part of the paper
* functions.py contains implementation of basic oracles for Nesterov's function and logistic regression
* utils.py contains functions for preparing data and plotting the results

## Jupyter Notebooks
Each .ipynb file corresponds to the particular set of experiments with given dimension of the problem (for Nesterov's function) or given dataset. All datasets are taken from [LIBSVM library](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).

## How to run the code
In order to run the code one needs to create folders "dump", "plot" and "datasets" in the same directory with jupyter notebooks, download corresponding datasets from LIBSVM library in txt-format, and put them in the folder "datasets". 
