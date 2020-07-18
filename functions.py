import numpy as np
from scipy.linalg import eigh as largest_eigh
from numpy import exp
import pickle
import random
from numpy.linalg import norm

from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.special import expit


def logreg_loss(x, args):
    A = args[0]
    y = args[1]
    l2 = args[2]
    sparse = args[3]
    assert l2 >= 0
    assert len(y) == A.shape[0]
    assert len(x) == A.shape[1]
    degree1 = np.zeros(A.shape[0])
    if sparse == True:
        degree2 = -(A * x) * y
        l = np.logaddexp(degree1, degree2)
    else:
        degree2 = -A.dot(x) * y
        l = np.logaddexp(degree1, degree2)
    m = y.shape[0]
    return np.sum(l) / m + l2/2 * norm(x) ** 2

def logreg_grad(x, args):
    A = args[0]
    y = args[1]
    mu = args[2]
    sparse = args[3]
    assert mu >= 0
    assert len(y) == A.shape[0]
    assert len(x) == A.shape[1]
    if sparse == True:
        degree = -y * (A * x)
        sigmas = expit(degree)
        loss_grad = -A.transpose() * (y * sigmas) / A.shape[0]
    else:
        degree = -y * (A.dot(x))
        sigmas = expit(degree)
        loss_grad = -A.T.dot(y * sigmas) / A.shape[0]
    assert len(loss_grad) == len(x)
    return loss_grad + mu * x


def logreg_sgrad(w, x_i, y_i):
    loss_sgrad = -y_i * x_i / (1 + np.exp(y_i * np.dot(x_i, w)))
    return loss_sgrad

def Nesterov_func(x, args):
    L = args[0]
    return ((np.sum(np.diff(x) ** 2) + x[0] ** 2 + x[x.shape[0] - 1] ** 2) / 2 - x[0]) * L / 4

def Nesterov_grad(x, args):
    n = x.shape[0]
    g = np.zeros(n)
    L = args[0]
    g[0] = L / 4 * (2 * x[0] - x[1] - 1)
    g[n - 1] = L / 4 * (2 * x[n - 1] - x[n - 2])
    for i in range(n - 2):
        g[i + 1] = L / 4 * (2 * x[i + 1] - x[i] - x[i + 2])
    return g

def logreg_plus_lasso(x, args):
    return logreg(x, args[0:2]) + lasso(x, args[2:4])

def logreg_plus_lasso_grad(x, args):
    return logreg_grad(x, args[0:2]) + lasso_grad(x, args[2:4])