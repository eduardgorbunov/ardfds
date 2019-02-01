import numpy as np
import pickle
from scipy.stats import norm


def sphere_point_generator(ndim):
    def sample_spherical(npoints):
        vec = norm().rvs(size=ndim)
        vec /= np.linalg.norm(vec)
        return np.array(vec)
    
    return lambda npoints: sample_spherical(npoints)


def gaussian_point_generator(ndim):
    def sample_gaussian(npoints):
        vec = norm().rvs(size=ndim)
        return np.array(vec)

    return lambda npoints: sample_gaussian(npoints)
