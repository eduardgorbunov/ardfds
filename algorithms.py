import numpy as np
#from functions import noisy_paired_values
from generator import sphere_point_generator
import time
import pickle
from scipy.stats import norm


def approximate_gradient(func, x, xi, direction_generator, t, m=1):
    direction = direction_generator(1)
    difference = 0.0
    for i in range(m):
        difference += func(x+t*direction, xi) - func(x, xi)
    
    gradient = direction * difference/(t*m)
    
    return gradient

def V_function(alpha, eg, z, x, order=0):
    z = np.asmatrix(z)
    x = np.asmatrix(x)
    n = x.shape[0]

    if order == 0:
        value = alpha * n * eg.T * (x - z) + 0.5 * (x - z).T * (x - z)
        return value
    if order == 1:
        value = alpha * n * eg.T * (x - z) + 0.5 * (x - z).T * (x - z)
        gradient = alpha * n * eg + x - z
        return (value, gradient)
    if order == 2:
        value = alpha * n * eg.T * (x - z) + 0.5 * (x - z).T * (x - z)
        gradient = alpha * n * eg + x - z
        hessian = np.identity(x.shape[0])
        return (value, gradient, hessian)


def mirror_descent(x, gradient, step_size):
    V = x - step_size * gradient
    return V

def gradient_kappa_norm_squared(x):
    n = len(x)
    grad = np.zeros(n)
    kappa = 1.0 + 1.0/np.log(n)
    sum_kappa_abs = np.sum(np.abs(x)**(kappa))
    grad = 2*np.power(sum_kappa_abs,2.0/kappa -1)*((np.abs(x)**(kappa-1))*np.sign(x))*1.0
    return grad

def mirror_descent_ne(x, gradient, step_size):
    n = len(x)
    alpha = step_size
    s = alpha*gradient
    kappa = 1.0 + 1.0/np.log(n)
    grad_norm = gradient_kappa_norm_squared(x)
    A_n = np.exp(1)*np.power(n,(kappa-1)*(2-kappa)*1.0/kappa)*np.log(n)/2
    s = grad_norm - s*1.0/A_n
    sum_hat_s = np.sum(np.abs(s*1.0/2)**(kappa*1.0/(kappa-1)))
    V = np.sign(s)*np.abs(s*1.0/2)**(1.0/(kappa-1))*np.power(sum_hat_s,(kappa-2)*1.0/kappa)
    return V

def ardfds_e(
    func,
    noisy_func,
    initial_x,
    L,
    m,
    t,
    f_star,
    filename,
    maximum_iterations=1000,
    direction_generator=None,
    sigma = 0.0001,
    stepsize_constant = 1
):
    x = initial_x.copy()
    y = x.copy()
    z = x.copy()
    n = len(x)
    # initialization
    
    # new   
    residuals_func = np.zeros((maximum_iterations + 1))*1.0
    start_residual_func = func(x) - f_star
    residuals_func[0] = 1.0
    
    runtimes = np.zeros((maximum_iterations + 1))*1.0
    start_time = time.time()
    runtimes[0] = (time.time() - start_time)
    
    for k in range(maximum_iterations):
        alpha = (k + 2) / (96 * n * n * L)
        tau = 2 / (k + 2)
        x = tau * z + (1 - tau) * y
        xi = norm(scale=sigma).rvs()
        gradient = approximate_gradient(noisy_func, x, xi, direction_generator, t, m)
        y = x - 1 / (2 * L) * gradient
        z = mirror_descent(z, gradient, alpha * n *stepsize_constant)
        residuals_func[k]=((func(y)-f_star)*1.0/start_residual_func)
        runtimes[k] = (time.time() - start_time)
    pickle.dump(runtimes, open("dump/" + filename + '_' + str(len(x)) + 'dim_' + str(maximum_iterations) + 'it_' + str(stepsize_constant) + 'stepsz' + "_runtimes_ardfds_e.txt", 'wb'))
    pickle.dump(residuals_func, open("dump/" + filename + '_' + str(len(x)) + 'dim_' + str(maximum_iterations) + 'it_' + str(stepsize_constant) + 'stepsz' + "_func_ardfds_e.txt", 'wb'))
    return y

def ardfds_ne(
    func,
    noisy_func,
    initial_x,
    L,
    m,
    t,
    f_star,
    filename,
    maximum_iterations=1000,
    direction_generator=None,
    sigma = 0.0001,
    stepsize_constant = 1
):
    x = initial_x.copy()
    y = x.copy()
    z = x.copy()
    n = len(x)

    residuals_func = np.zeros((maximum_iterations + 1))*1.0
    start_residual_func = func(x) - f_star
    residuals_func[0] = 1.0
    
    runtimes = np.zeros((maximum_iterations + 1))*1.0
    start_time = time.time()
    runtimes[0] = (time.time() - start_time)
    if not direction_generator:
        direction_generator = sphere_point_generator(n)

    rho_n = (16.0*np.log(n) - 8.0)/n
    for k in range(maximum_iterations):
        alpha = (k + 2) / (96 * n * (16.0*np.log(n) - 8.0) * L)
        tau = 2 / (k + 2)
        x = tau * z + (1 - tau) * y
        xi = norm(scale=sigma).rvs()
        gradient = approximate_gradient(noisy_func, x, xi, direction_generator, t, m)
        y = x - 1 / (2 * L) * gradient
        z = mirror_descent_ne(z, gradient, alpha * n * stepsize_constant)
        residuals_func[k] = ((func(y)-f_star)*1.0/start_residual_func)
        runtimes[k] = (time.time() - start_time)
    pickle.dump(runtimes, open("dump/" + filename + '_' + str(len(x)) + 'dim_' + str(maximum_iterations) + 'it_' + str(stepsize_constant) + 'stepsz' + "_runtimes_ardfds_ne.txt", 'wb'))
    pickle.dump(residuals_func, open("dump/" + filename + '_' + str(len(x)) + 'dim_' + str(maximum_iterations) + 'it_' + str(stepsize_constant) + 'stepsz' + "_func_ardfds_ne.txt", 'wb'))
    return y

def rdfds_e(
    func,
    noisy_func,
    initial_x,
    L,
    m,
    t,
    f_star,
    filename,
    maximum_iterations=1000,
    direction_generator=None,
    sigma = 0.0001,
    stepsize_constant = 1
):
    
    x = initial_x.copy()
    n = len(x)
    
    residuals_func = np.zeros((maximum_iterations + 1))*1.0
    start_residual_func = func(x) - f_star
    residuals_func[0] = 1.0
    
    runtimes = np.zeros((maximum_iterations + 1))*1.0
    start_time = time.time()
    runtimes[0] = (time.time() - start_time)

    if not direction_generator:
        direction_generator = sphere_point_generator(n)

    for k in range(maximum_iterations):
        alpha = 1 / (48 * n * L)
        xi = norm(scale=sigma).rvs()
        gradient = approximate_gradient(noisy_func, x, xi, direction_generator, t, m)
        x = mirror_descent(x, gradient, alpha * n * stepsize_constant)
        residuals_func[k] = ((func(x)-f_star)*1.0/start_residual_func)
        runtimes[k] = (time.time() - start_time)
    pickle.dump(runtimes, open("dump/" + filename + '_' + str(len(x)) + 'dim_' + str(maximum_iterations) + 'it_' + str(stepsize_constant) + 'stepsz' + "_runtimes_rdfds_e.txt", 'wb'))
    pickle.dump(residuals_func, open("dump/" + filename + '_' + str(len(x)) + 'dim_' + str(maximum_iterations) + 'it_' + str(stepsize_constant) + 'stepsz' + "_func_rdfds_e.txt", 'wb'))
    return x

def rdfds_ne(
    func,
    noisy_func,
    initial_x,
    L,
    m,
    t,
    f_star,
    filename,
    maximum_iterations=1000,
    direction_generator=None,
    sigma = 0.0001,
    stepsize_constant = 1
):

    x = initial_x.copy()
    n = len(x)

    residuals_func = np.zeros((maximum_iterations + 1))*1.0
    start_residual_func = func(x) - f_star
    residuals_func[0] = 1.0
    
    runtimes = np.zeros((maximum_iterations + 1))*1.0
    start_time = time.time()
    runtimes[0] = (time.time() - start_time)

    if not direction_generator:
        direction_generator = sphere_point_generator(n)

    rho_n = (16.0*np.log(n) - 8.0)/n
    for k in range(maximum_iterations):
        alpha = 1 / (48 * n *rho_n * L)
        xi = norm(scale=sigma).rvs()
        gradient = approximate_gradient(noisy_func, x, xi, direction_generator, t, m)
        x = mirror_descent_ne(x, gradient, alpha * n * stepsize_constant)
        residuals_func[k] = ((func(x)-f_star)*1.0/start_residual_func)
        runtimes[k] = (time.time() - start_time)
    pickle.dump(runtimes, open("dump/" + filename + '_' + str(len(x)) + 'dim_' + str(maximum_iterations) + 'it_' + str(stepsize_constant) + 'stepsz' + "_runtimes_rdfds_ne.txt", 'wb'))
    pickle.dump(residuals_func, open("dump/" + filename + '_' + str(len(x)) + 'dim_' + str(maximum_iterations) + 'it_' + str(stepsize_constant) + 'stepsz' + "_func_rdfds_ne.txt", 'wb'))
    return x

def rsgf(
    func,
    noisy_func,
    initial_x,
    L,
    m,
    mu,
    f_star,
    filename,
    maximum_iterations=1000,
    initial_stepsize=1,
    direction_generator=None,
    sigma = 0.0001
):
  
    x = initial_x.copy()
    n = len(x)

    residuals_func = np.zeros((maximum_iterations + 1))*1.0
    start_residual_func = func(x) - f_star
    residuals_func[0] = 1.0
    
    runtimes = np.zeros((maximum_iterations + 1))*1.0
    start_time = time.time()
    runtimes[0] = (time.time() - start_time)

    if not direction_generator:
        direction_generator = sphere_point_generator(n)

    h = 1 / np.sqrt(n + 4) * min(
        1 / (4 * L * np.sqrt(n + 4)),
        initial_stepsize / np.sqrt(maximum_iterations)
    )

    for k in range(maximum_iterations):
        xi = norm(scale=sigma).rvs()
        gradient = approximate_gradient(noisy_func, x, xi, direction_generator, mu, 1)
        x = x - h * gradient
        residuals_func[k] = ((func(x)-f_star)*1.0/start_residual_func)
        runtimes[k] = (time.time() - start_time)
    pickle.dump(runtimes, open("dump/" + filename + '_' + str(len(x)) + 'dim_' + str(maximum_iterations) + 'it' + "_runtimes_rsgf.txt", 'wb'))
    pickle.dump(residuals_func, open("dump/" + filename + '_' + str(len(x)) + 'dim_' + str(maximum_iterations) + 'it' + "_func_rsgf.txt", 'wb'))
    return x