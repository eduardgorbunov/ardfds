import numpy as np
import time
import pickle
from scipy.stats import norm
from copy import deepcopy
from copy import copy
from scipy.stats import multivariate_normal as mltnorm
from scipy.stats import randint
import scipy


def ardfds_e_noise_logreg_full(filename, x_init, args, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    dumping_constant = np.max([int(N/(10000)), 1])
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    yk = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    alpha_coeff = (1.0 / (96*(n**2)*L)) * tuning_stepsize_param
    
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(N):
        tau = 2.0 / (k+2)
        alpha = (k+2) * alpha_coeff
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        x = tau*z + (1-tau)*yk
        tnabla = e * (f(x+t*e,[A_for_batch, y, l2, sparse]) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,[A_for_batch, y, l2, sparse]) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        yk = x - tnabla * 0.5 / L
        z = z - tnabla * n * alpha
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(yk, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*m)
            
    res = {'last_iter'   : yk, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_ARDFDS_E_logreg_full_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def ardfds_ne_noise_logreg_full(filename, x_init, args, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N/(10000)), 1])
    
    yk = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    kappa = 1.0 + 1.0/np.log(n)
    A_n = np.exp(1)*np.power(n,(kappa-1)*(2-kappa)*1.0/kappa)*np.log(n)/2
    
    alpha_coef = (1.0 / (96*(n*(16* np.log(n) - 8))*L)) * tuning_stepsize_param
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(N):
        tau = 2.0 / (k+2)
        alpha = (k+2) * alpha_coef
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        x = tau*z + (1-tau)*yk
        tnabla = e * (f(x+t*e,[A_for_batch, y, l2, sparse]) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,[A_for_batch, y, l2, sparse]) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        yk = x - tnabla * 0.5 / L
        s = alpha*n*tnabla
        grad_norm = gradient_kappa_norm_squared(z)
        s = grad_norm - s*1.0/A_n
        sum_hat_s = np.sum(np.abs(s*1.0/2)**(kappa*1.0/(kappa-1)))
        z = np.sign(s)*np.abs(s*1.0/2)**(1.0/(kappa-1))*np.power(sum_hat_s,(kappa-2)*1.0/kappa)
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(yk, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*m)
            
    res = {'last_iter'   : yk, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_ARDFDS_NE_logreg_full_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res

def rdfds_e_noise_logreg_full(filename, x_init, args, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N/(10000)), 1])
    
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    alpha = (1.0 / (48*n*L)) * tuning_stepsize_param
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(N):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        tnabla = e * (f(x+t*e,[A_for_batch, y, l2, sparse]) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,[A_for_batch, y, l2, sparse]) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        x = x - tnabla * n * alpha
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*m)
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RDFDS_E_logreg_full_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res

def rdfds_ne_noise_logreg_full(filename, x_init, args, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N/(10000)), 1])
    
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    kappa = 1.0 + 1.0/np.log(n)
    A_n = np.exp(1)*np.power(n,(kappa-1)*(2-kappa)*1.0/kappa)*np.log(n)/2
    
    alpha = (1.0 / (48*(16* np.log(n) - 8)*L)) * tuning_stepsize_param
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(N):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        tnabla = e * (f(x+t*e,[A_for_batch, y, l2, sparse]) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,[A_for_batch, y, l2, sparse]) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        
        s = alpha*n*tnabla
        grad_norm = gradient_kappa_norm_squared(x)
        s = grad_norm - s*1.0/A_n
        sum_hat_s = np.sum(np.abs(s*1.0/2)**(kappa*1.0/(kappa-1)))
        x = np.sign(s)*np.abs(s*1.0/2)**(1.0/(kappa-1))*np.power(sum_hat_s,(kappa-2)*1.0/kappa)
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*m)
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RDFDS_NE_logreg_full_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def rsgf_tune_noise_logreg_full(filename, x_init, args, N=100, f_star=None, x_star=None, initial_stepsize=1.0, tuning_stepsize=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N/(10000)), 1])
    
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0

    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    h = 1.0 / np.sqrt(n + 4) * min(
        1.0 / (4 * L * np.sqrt(n + 4)),
        initial_stepsize * 1.0 / np.sqrt(int(N))
    )*tuning_stepsize
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(N):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        e = temp_arr[directions_counter*n:(directions_counter+1)*n]
        directions_counter += 1
        
        tnabla = e * (f(x+t*e,[A_for_batch, y, l2, sparse]) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,[A_for_batch, y, l2, sparse]) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        x = x - tnabla * h
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*m)
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RSGF_logreg_full_noise_steps_const_"+str(initial_stepsize)+"_tun_steps_"+str(tuning_stepsize)
              +"_epochs_"+str(N)+"_delta_"+str(delta)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def ardfds_e_noise_logreg(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N*m/(bs*10000)), 1])
    
    yk = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N*m*1.0/bs), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    indices = randint.rvs(low=0, high=m, size=min(int(N*m*1.0/bs), int(100000/bs))*bs)
    indices_size = len(indices)
    indices_counter = 0
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    alpha_coeff = (1.0 / (96*(n**2)*L)) * tuning_stepsize_param
    
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(int(N*m*1.0/bs)):
        tau = 2.0 / (k+2)
        alpha = (k+2) * alpha_coeff
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        
        batch_ind = indices[indices_counter:(indices_counter+bs)]
        indices_counter += bs
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        x = tau*z + (1-tau)*yk
        tnabla = e * (f(x+t*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        yk = x - tnabla * 0.5 / L
        z = z - tnabla * n * alpha
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(yk, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*bs)
            
    res = {'last_iter'   : yk, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_ARDFDS_E_logreg_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+"_batch_"+str(bs)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def ardfds_ne_noise_logreg(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N*m/(bs*10000)), 1])
    
    yk = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N*m*1.0/bs), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    indices = randint.rvs(low=0, high=m, size=min(int(N*m*1.0/bs), int(100000/bs))*bs)
    indices_size = len(indices)
    indices_counter = 0
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    kappa = 1.0 + 1.0/np.log(n)
    A_n = np.exp(1)*np.power(n,(kappa-1)*(2-kappa)*1.0/kappa)*np.log(n)/2
    
    alpha_coef = (1.0 / (96*(n*(16* np.log(n) - 8))*L)) * tuning_stepsize_param
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(int(N*m*1.0/bs)):
        tau = 2.0 / (k+2)
        alpha = (k+2) * alpha_coef
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        
        batch_ind = indices[indices_counter:(indices_counter+bs)]
        indices_counter += bs
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        x = tau*z + (1-tau)*yk
        tnabla = e * (f(x+t*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        yk = x - tnabla * 0.5 / L
        s = alpha*n*tnabla
        grad_norm = gradient_kappa_norm_squared(z)
        s = grad_norm - s*1.0/A_n
        sum_hat_s = np.sum(np.abs(s*1.0/2)**(kappa*1.0/(kappa-1)))
        z = np.sign(s)*np.abs(s*1.0/2)**(1.0/(kappa-1))*np.power(sum_hat_s,(kappa-2)*1.0/kappa)
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(yk, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*bs)
            
    res = {'last_iter'   : yk, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_ARDFDS_NE_logreg_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+"_batch_"+str(bs)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def rdfds_e_noise_logreg(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N*m/(bs*10000)), 1])
    
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N*m*1.0/bs), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    indices = randint.rvs(low=0, high=m, size=min(int(N*m*1.0/bs), int(100000/bs))*bs)
    indices_size = len(indices)
    indices_counter = 0
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    alpha = (1.0 / (48*n*L)) * tuning_stepsize_param
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(int(N*m*1.0/bs)):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        
        batch_ind = indices[indices_counter:(indices_counter+bs)]
        indices_counter += bs
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        tnabla = e * (f(x+t*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        x = x - tnabla * n * alpha
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*bs)
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RDFDS_E_logreg_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+"_batch_"+str(bs)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def rdfds_ne_noise_logreg(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N*m/(bs*10000)), 1])
    
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N*m*1.0/bs), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    indices = randint.rvs(low=0, high=m, size=min(int(N*m*1.0/bs), int(100000/bs))*bs)
    indices_size = len(indices)
    indices_counter = 0
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    kappa = 1.0 + 1.0/np.log(n)
    A_n = np.exp(1)*np.power(n,(kappa-1)*(2-kappa)*1.0/kappa)*np.log(n)/2
    
    alpha = (1.0 / (48*(16* np.log(n) - 8)*L)) * tuning_stepsize_param
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(int(N*m*1.0/bs)):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        
        batch_ind = indices[indices_counter:(indices_counter+bs)]
        indices_counter += bs
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        tnabla = e * (f(x+t*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        
        s = alpha*n*tnabla
        grad_norm = gradient_kappa_norm_squared(x)
        s = grad_norm - s*1.0/A_n
        sum_hat_s = np.sum(np.abs(s*1.0/2)**(kappa*1.0/(kappa-1)))
        x = np.sign(s)*np.abs(s*1.0/2)**(1.0/(kappa-1))*np.power(sum_hat_s,(kappa-2)*1.0/kappa)
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*bs)
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RDFDS_NE_logreg_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+"_batch_"+str(bs)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res

def rsgf_tune_noise_logreg(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, initial_stepsize=1.0, tuning_stepsize=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N*m/(bs*10000)), 1])
    
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N*m*1.0/bs), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0

    indices = randint.rvs(low=0, high=m, size=min(int(N*m*1.0/bs), int(100000/bs))*bs)
    indices_size = len(indices)
    indices_counter = 0
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    h = 1.0 / np.sqrt(n + 4) * min(
        1.0 / (4 * L * np.sqrt(n + 4)),
        initial_stepsize * 1.0 / np.sqrt(int(N*m*1.0/bs))
    )*tuning_stepsize
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(int(N*m*1.0/bs)):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        
        batch_ind = indices[indices_counter:(indices_counter+bs)]
        indices_counter += bs
        
        e = temp_arr[directions_counter*n:(directions_counter+1)*n]
        directions_counter += 1
        
        tnabla = e * (f(x+t*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        x = x - tnabla * h
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*bs)
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RSGF_logreg_noise_steps_const_"+str(initial_stepsize)+"_tun_steps_"+str(tuning_stepsize)
              +"_epochs_"+str(N)+"_delta_"+str(delta)+"_batch_"+str(bs)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def ardfds_e_noise(filename, x_init, a, args, m=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    dumping_constant = np.max([int(N/10000), 1])
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    L = args[1]
    sigma = args[2]
    delta = args[3]
    t = args[-1]
    y = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    number_of_xis = min(N*m, 10000*m)
    xis_arr = norm(scale=sigma).rvs(size=number_of_xis)
    
    xis_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    alpha_coeff = (1.0 / (96*(n**2)*L)) * tuning_stepsize_param
    
    for k in range(N):
        tau = 2.0 / (k+2)
        alpha = (k+2) * alpha_coeff
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if xis_counter == number_of_xis-m:
            xis_arr = norm(scale=sigma).rvs(size=number_of_xis)
            xis_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        xi = np.mean(xis_arr[xis_counter:xis_counter+m])
        xis_counter += m
        
        x = tau*z + (1-tau)*y
        tnabla = e * (f(x+t*e,args[1:]) + xi*np.dot(a,x+t*e) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,args[1:]) - xi*np.dot(a,x) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        y = x - tnabla * 0.5 / L
        z = z - tnabla * n * alpha
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(y,args[1:]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*m)
            
    res = {'last_iter'   : y, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_ARDFDS_E_noise_steps_const_"+str(tuning_stepsize_param)+"_iters_"+str(N)+
              "_sigma_"+str(sigma)+"_delta_"+str(delta)+"_batch_"+str(m)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def ardfds_ne_noise(filename, x_init, a, args, m=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    dumping_constant = np.max([int(N/10000), 1])
    
    f = args[0]
    L = args[1]
    sigma = args[2]
    delta = args[3]
    t = args[-1]
    y = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    directions_counter = 0
    
    number_of_xis = min(N, 10000)*m
    xis_arr = norm(scale=sigma).rvs(size=number_of_xis)
    
    xis_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    kappa = 1.0 + 1.0/np.log(n)
    A_n = np.exp(1)*np.power(n,(kappa-1)*(2-kappa)*1.0/kappa)*np.log(n)/2
    
    alpha_coef = (1.0 / (96*(n*(16* np.log(n) - 8))*L)) * tuning_stepsize_param
    
    for k in range(N):
        tau = 2.0 / (k+2)
        alpha = (k+2) * alpha_coef

        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if xis_counter == number_of_xis-m:
            xis_arr = norm(scale=sigma).rvs(size=number_of_xis)
            xis_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        xi = np.mean(xis_arr[xis_counter:xis_counter+m])
        xis_counter += m
        
        x = tau*z + (1-tau)*y
        tnabla = e * (f(x+t*e,args[1:]) + xi*np.dot(a,x+t*e) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,args[1:]) - xi*np.dot(a,x) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        y = x - tnabla * 0.5 / L
        
        s = alpha*n*tnabla
        grad_norm = gradient_kappa_norm_squared(z)
        s = grad_norm - s*1.0/A_n
        sum_hat_s = np.sum(np.abs(s*1.0/2)**(kappa*1.0/(kappa-1)))
        z = np.sign(s)*np.abs(s*1.0/2)**(1.0/(kappa-1))*np.power(sum_hat_s,(kappa-2)*1.0/kappa)
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(y,args[1:]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*m)
        
    res = {'last_iter'   : y, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_ARDFDS_NE_noise_steps_const_"+str(tuning_stepsize_param)+"_iters_"+str(N)+
              "_sigma_"+str(sigma)+"_delta_"+str(delta)+"_batch_"+str(m)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def rdfds_e_noise(filename, x_init, a, args, m=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    dumping_constant = np.max([int(N/10000), 1])
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    L = args[1]
    sigma = args[2]
    delta = args[3]
    t = args[-1]
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    number_of_xis = min(N, 10000)*m
    xis_arr = norm(scale=sigma).rvs(size=number_of_xis)
    
    xis_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    alpha = (1.0 / (48*n*L)) * tuning_stepsize_param
    
    for k in range(N):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if xis_counter == number_of_xis-m:
            xis_arr = norm(scale=sigma).rvs(size=number_of_xis)
            xis_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        xi = np.mean(xis_arr[xis_counter:xis_counter+m])
        xis_counter += m
        
        tnabla = e * (f(x+t*e,args[1:]) + xi*np.dot(a,x+t*e) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,args[1:]) - xi*np.dot(a,x) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        x = x - tnabla * n * alpha
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*m)
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RDFDS_E_noise_steps_const_"+str(tuning_stepsize_param)+"_iters_"+str(N)+
              "_sigma_"+str(sigma)+"_delta_"+str(delta)+"_batch_"+str(m)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res

def rdfds_ne_noise(filename, x_init, a, args, m=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    dumping_constant = np.max([int(N/10000), 1])
    
    f = args[0]
    L = args[1]
    sigma = args[2]
    delta = args[3]
    t = args[-1]
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    directions_counter = 0
    
    number_of_xis = min(N, 10000)*m
    xis_arr = norm(scale=sigma).rvs(size=number_of_xis)
    
    xis_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    kappa = 1.0 + 1.0/np.log(n)
    A_n = np.exp(1)*np.power(n,(kappa-1)*(2-kappa)*1.0/kappa)*np.log(n)/2
    
    alpha = (1.0 / (48*(16* np.log(n) - 8)*L)) * tuning_stepsize_param
    
    for k in range(N):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if xis_counter == number_of_xis-m:
            xis_arr = norm(scale=sigma).rvs(size=number_of_xis)
            xis_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        xi = np.mean(xis_arr[xis_counter:xis_counter+m])
        xis_counter += m
        
        tnabla = e * (f(x+t*e,args[1:]) + xi*np.dot(a,x+t*e) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,args[1:]) - xi*np.dot(a,x) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        
        s = alpha*n*tnabla
        grad_norm = gradient_kappa_norm_squared(x)
        s = grad_norm - s*1.0/A_n
        sum_hat_s = np.sum(np.abs(s*1.0/2)**(kappa*1.0/(kappa-1)))
        x = np.sign(s)*np.abs(s*1.0/2)**(1.0/(kappa-1))*np.power(sum_hat_s,(kappa-2)*1.0/kappa)
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*m)
        
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RDFDS_NE_noise_steps_const_"+str(tuning_stepsize_param)+"_iters_"+str(N)+
              "_sigma_"+str(sigma)+"_delta_"+str(delta)+"_batch_"+str(m)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def rsgf_tune_noise(filename, x_init, a, args, m=1, N=100, f_star=None, x_star=None, initial_stepsize=1.0, tuning_stepsize=1.0):
    n = len(x_init)
    dumping_constant = np.max([int(N/10000), 1])
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    L = args[1]
    sigma = args[2]
    delta = args[3]
    t = args[-1]
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0

    number_of_xis = min(N, 10000)*m
    xis_arr = norm(scale=sigma).rvs(size=number_of_xis)
    
    xis_counter = 0
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    h = 1.0 / np.sqrt(n + 4) * min(
        1.0 / (4 * L * np.sqrt(n + 4)),
        initial_stepsize * 1.0 / np.sqrt(N)
    )*tuning_stepsize
    
    for k in range(N):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if xis_counter == number_of_xis-m:
            xis_arr = norm(scale=sigma).rvs(size=number_of_xis)
            xis_counter = 0
        
        e = temp_arr[directions_counter*n:(directions_counter+1)*n]
        directions_counter += 1
        
        xi = np.mean(xis_arr[xis_counter:xis_counter+m])
        xis_counter += m
        
        tnabla = e * (f(x+t*e,args[1:]) + xi*np.dot(a,x+t*e) + delta*np.sin(1.0/(np.linalg.norm(x+t*e-x_star)**2)) - f(x,args[1:]) - xi*np.dot(a,x) - delta*np.sin(1.0/(np.linalg.norm(x-x_star)**2))) * 1.0/t
        x = x - tnabla * h
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*m)
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RSGF_noise_steps_const_"+str(initial_stepsize)+"_tun_steps_"+str(tuning_stepsize)
              +"_iters_"+str(N)+"_sigma_"+str(sigma)+"_delta_"+str(delta)+"_batch_"+str(m)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def ardfds_e(filename, x_init, args, N = 100, f_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    dumping_constant = np.max([int(N/10000), 1])
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    L = args[1]
    t = args[-1]
    y = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    alpha_coeff = (1.0 / (96*(n**2)*L)) * tuning_stepsize_param
    
    for k in range(N):
        tau = 2.0 / (k+2)
        alpha = (k+2) * alpha_coeff
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        x = tau*z + (1-tau)*y
        tnabla = e * (f(x+t*e,args[1:]) - f(x,args[1:])) * 1.0/t
        y = x - tnabla * 0.5 / L
        z = z - tnabla * n * alpha
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(y,args[1:]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2)
            
    res = {'last_iter'   : y, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_ARDFDS_E_steps_const_"+str(tuning_stepsize_param)+"_iters_"+str(N)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res
            
        

def ardfds_ne(filename, x_init, args, N = 100, f_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    dumping_constant = np.max([int(N/10000), 1])
    
    f = args[0]
    L = args[1]
    t = args[-1]
    y = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    directions_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    kappa = 1.0 + 1.0/np.log(n)
    A_n = np.exp(1)*np.power(n,(kappa-1)*(2-kappa)*1.0/kappa)*np.log(n)/2
    
    alpha_coef = (1.0 / (96*(n*(16* np.log(n) - 8))*L)) * tuning_stepsize_param
    
    for k in range(N):
        tau = 2.0 / (k+2)
        alpha = (k+2) * alpha_coef

        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        x = tau*z + (1-tau)*y
        tnabla = e * (f(x+t*e,args[1:]) - f(x,args[1:])) * 1.0/t
        y = x - tnabla * 0.5 / L
        
        s = alpha*n*tnabla
        grad_norm = gradient_kappa_norm_squared(z)
        s = grad_norm - s*1.0/A_n
        sum_hat_s = np.sum(np.abs(s*1.0/2)**(kappa*1.0/(kappa-1)))
        z = np.sign(s)*np.abs(s*1.0/2)**(1.0/(kappa-1))*np.power(sum_hat_s,(kappa-2)*1.0/kappa)
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(y,args[1:]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2)
        
    res = {'last_iter'   : y, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_ARDFDS_NE_steps_const_"+str(tuning_stepsize_param)+"_iters_"+str(N)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res




def rdfds_e(filename, x_init, args, N = 100, f_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    dumping_constant = np.max([int(N/10000), 1])
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    L = args[1]
    t = args[-1]
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    alpha = (1.0 / (48*n*L)) * tuning_stepsize_param
    
    for k in range(N):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        tnabla = e * (f(x+t*e,args[1:]) - f(x,args[1:])) * 1.0/t
        x = x - tnabla * n * alpha
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2)
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RDFDS_E_steps_const_"+str(tuning_stepsize_param)+"_iters_"+str(N)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res
            
        

def rdfds_ne(filename, x_init, args, N = 100, f_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    dumping_constant = np.max([int(N/10000), 1])
    
    f = args[0]
    L = args[1]
    t = args[-1]
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    directions_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    kappa = 1.0 + 1.0/np.log(n)
    A_n = np.exp(1)*np.power(n,(kappa-1)*(2-kappa)*1.0/kappa)*np.log(n)/2
    
    alpha = (1.0 / (48*(16* np.log(n) - 8)*L)) * tuning_stepsize_param
    
    for k in range(N):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        tnabla = e * (f(x+t*e,args[1:]) - f(x,args[1:])) * 1.0/t
        
        s = alpha*n*tnabla
        grad_norm = gradient_kappa_norm_squared(x)
        s = grad_norm - s*1.0/A_n
        sum_hat_s = np.sum(np.abs(s*1.0/2)**(kappa*1.0/(kappa-1)))
        x = np.sign(s)*np.abs(s*1.0/2)**(1.0/(kappa-1))*np.power(sum_hat_s,(kappa-2)*1.0/kappa)
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2)
        
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RDFDS_NE_steps_const_"+str(tuning_stepsize_param)+"_iters_"+str(N)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res



def rsgf_tune(filename, x_init, args, N = 100, f_star=None, initial_stepsize=1.0, tuning_stepsize=1.0):
    n = len(x_init)
    dumping_constant = np.max([int(N/10000), 1])
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    L = args[1]
    t = args[-1]
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    h = 1.0 / np.sqrt(n + 4) * min(
        1.0 / (4 * L * np.sqrt(n + 4)),
        initial_stepsize * 1.0 / np.sqrt(N)
    )*tuning_stepsize
    
    for k in range(N):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        e = temp_arr[directions_counter*n:(directions_counter+1)*n]
        directions_counter += 1
        
        tnabla = e * (f(x+t*e,args[1:]) - f(x,args[1:])) * 1.0/t
        x = x - tnabla * h
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2)
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RSGF_steps_const_"+str(initial_stepsize)+"_tun_steps_"+str(tuning_stepsize)
              +"_iters_"+str(N)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res



def rsgf(filename, x_init, args, N = 100, f_star=None, initial_stepsize=1.0):
    n = len(x_init)
    dumping_constant = np.max([int(N/10000), 1])
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    L = args[1]
    t = args[-1]
    x = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    h = 1.0 / np.sqrt(n + 4) * min(
        1.0 / (4 * L * np.sqrt(n + 4)),
        initial_stepsize * 1.0 / np.sqrt(N)
    )
    
    for k in range(N):
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        e = temp_arr[directions_counter*n:(directions_counter+1)*n]
        directions_counter += 1
        
        tnabla = e * (f(x+t*e,args[1:]) - f(x,args[1:])) * 1.0/t
        x = x - tnabla * h
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x,args[1:]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2)
            
    res = {'last_iter'   : x, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RSGF_steps_const_"+str(initial_stepsize)+"_iters_"+str(N)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res



def gradient_kappa_norm_squared(x):
    n = len(x)
    grad = np.zeros(n)
    kappa = 1.0 + 1.0/np.log(n)
    sum_kappa_abs = np.sum(np.abs(x)**(kappa))
    grad = 2*np.power(sum_kappa_abs,2.0/kappa -1)*((np.abs(x)**(kappa-1))*np.sign(x))*1.0
    return grad