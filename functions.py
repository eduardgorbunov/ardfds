import numpy as np
from scipy.linalg import eigh as largest_eigh
from numpy import exp
import pickle

def quadratic(H, b, x, order=0):
    value = 0.5 * np.dot(x,np.dot(H,x)) + np.dot(b,x)
    return value

def nesterov_function(x):
    n = len(x)
    value = 0.5*x[0]*x[0] + 0.5*x[n-1]*x[n-1] - x[0]
    for i in range(n-1):
        value += 0.5*(x[i+1] - x[i])*(x[i+1] - x[i])
    return value*0.25

def noise(x, xi, a, Delta):
    return xi*np.dot(x,a) + Delta*np.sin(np.linalg.norm(x))
