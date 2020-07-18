import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib
import pickle
from scipy.stats import randint
from scipy.stats import uniform
from scipy.optimize import minimize
import copy
import math
import time
from scipy.optimize import minimize
from scipy.sparse.linalg import svds
from scipy.linalg import svdvals
import scipy
from sklearn.datasets import load_svmlight_file
import pickle
from pathlib import Path

def prepare_data(dataset):
    filename = "datasets/" + dataset + ".txt"

    data = load_svmlight_file(filename)
    A, y = data[0], data[1]
    m, n = A.shape
    
    if (2 in y) & (1 in y):
        y = 2 * y - 3
    if (2 in y) & (4 in y):
        y = y - 3
    if (0 in y) & (1 in y):
        y = 2*y - 1
    assert((-1 in y) & (1 in y))
    
    sparsity_A = A.count_nonzero() / (m * n)
    return A, y, m, n, sparsity_A


def compute_L(dataset, A):
    filename = "dump/"+dataset+"_L.txt"
    file_path = Path(filename)
    if file_path.is_file():
        with open(filename, 'rb') as file:
            L, average_L, worst_L = pickle.load(file)
    else:
        sigmas = svds(A, return_singular_vectors=False)
        m = A.shape[0]
        L = sigmas.max()**2 / (4*m)
        
        worst_L = 0
        average_L = 0
        denseA = A.toarray()
        for i in range(m):
            L_temp = (np.linalg.norm(denseA[i])**2)*1.0 / 4
            average_L += L_temp / m
            if L_temp > worst_L:
                worst_L = L_temp
        with open(filename, 'wb') as file:
            pickle.dump([L, average_L, worst_L],file)
    return L, average_L, worst_L


def save_solution(dataset, l2, l1, x_star, f_star):
    filename = "dump/"+dataset+"_solution_l2_"+str(l2)+"_l1_"+str(l1)+".txt"
    with open(filename, 'wb') as file:
        pickle.dump([x_star, f_star], file)

def read_solution(dataset, l2, l1):
    with open('dump/'+dataset+'_solution_l2_'+str(l2)+"_l1_"+str(l1)+".txt", 'rb') as file:
        return pickle.load(file)

def read_results_from_file(filename, method, args):
    if method == 'ARDFDS_E':
        with open('dump/'+filename+"_ARDFDS_E_steps_const_"+str(args[0])
                  +'_iters_'+str(args[1])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'ARDFDS_E noise':
        with open('dump/'+filename+"_ARDFDS_E_noise_steps_const_"+str(args[0])
                  +'_iters_'+str(args[1])+"_sigma_"+str(args[2])+"_delta_"+str(args[3])
                  +"_batch_"+str(args[4])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'ARDFDS_E logreg':
        with open('dump/'+filename+"_ARDFDS_E_logreg_noise_steps_const_"+str(args[0])
                  +'_epochs_'+str(args[1])+"_delta_"+str(args[2])
                  +"_batch_"+str(args[3])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'ARDFDS_E full':
        with open('dump/'+filename+"_ARDFDS_E_logreg_full_noise_steps_const_"+str(args[0])
                  +'_epochs_'+str(args[1])+"_delta_"+str(args[2])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'ARDFDS_NE':
        with open('dump/'+filename+"_ARDFDS_NE_steps_const_"+str(args[0])
                  +'_iters_'+str(args[1])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'ARDFDS_NE noise':
        with open('dump/'+filename+"_ARDFDS_NE_noise_steps_const_"+str(args[0])
                  +'_iters_'+str(args[1])+"_sigma_"+str(args[2])+"_delta_"+str(args[3])
                  +"_batch_"+str(args[4])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'ARDFDS_NE logreg':
        with open('dump/'+filename+"_ARDFDS_NE_logreg_noise_steps_const_"+str(args[0])
                  +'_epochs_'+str(args[1])+"_delta_"+str(args[2])
                  +"_batch_"+str(args[3])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'ARDFDS_NE full':
        with open('dump/'+filename+"_ARDFDS_NE_logreg_full_noise_steps_const_"+str(args[0])
                  +'_epochs_'+str(args[1])+"_delta_"+str(args[2])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RDFDS_E':
        with open('dump/'+filename+"_RDFDS_E_steps_const_"+str(args[0])
                  +'_iters_'+str(args[1])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RDFDS_E noise':
        with open('dump/'+filename+"_RDFDS_E_noise_steps_const_"+str(args[0])
                  +'_iters_'+str(args[1])+"_sigma_"+str(args[2])+"_delta_"+str(args[3])
                  +"_batch_"+str(args[4])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RDFDS_E logreg':
        with open('dump/'+filename+"_RDFDS_E_logreg_noise_steps_const_"+str(args[0])
                  +'_epochs_'+str(args[1])+"_delta_"+str(args[2])
                  +"_batch_"+str(args[3])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RDFDS_E full':
        with open('dump/'+filename+"_RDFDS_E_logreg_full_noise_steps_const_"+str(args[0])
                  +'_epochs_'+str(args[1])+"_delta_"+str(args[2])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RDFDS_NE noise':
        with open('dump/'+filename+"_RDFDS_NE_noise_steps_const_"+str(args[0])
                  +'_iters_'+str(args[1])+"_sigma_"+str(args[2])+"_delta_"+str(args[3])
                  +"_batch_"+str(args[4])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RDFDS_NE logreg':
        with open('dump/'+filename+"_RDFDS_NE_logreg_noise_steps_const_"+str(args[0])
                  +'_epochs_'+str(args[1])+"_delta_"+str(args[2])
                  +"_batch_"+str(args[3])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RDFDS_NE full':
        with open('dump/'+filename+"_RDFDS_NE_logreg_full_noise_steps_const_"+str(args[0])
                  +'_epochs_'+str(args[1])+"_delta_"+str(args[2])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RDFDS_NE':
        with open('dump/'+filename+"_RDFDS_NE_steps_const_"+str(args[0])
                  +'_iters_'+str(args[1])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RSGF':
        with open('dump/'+filename+"_RSGF_steps_const_"+str(args[0])
                  +'_iters_'+str(args[1])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RSGF tune':
        with open('dump/'+filename+"_RSGF_steps_const_"+str(args[0])+"_tun_steps_"+str(args[1])
                  +'_iters_'+str(args[2])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RSGF tune noise':
        with open('dump/'+filename+"_RSGF_noise_steps_const_"+str(args[0])+"_tun_steps_"+str(args[1])
                  +'_iters_'+str(args[2])+"_sigma_"+str(args[3])+"_delta_"+str(args[4])
                  +"_batch_"+str(args[5])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RSGF logreg':
        with open('dump/'+filename+"_RSGF_logreg_noise_steps_const_"+str(args[0])+"_tun_steps_"+str(args[1])
                  +'_epochs_'+str(args[2])+"_delta_"+str(args[3])
                  +"_batch_"+str(args[4])+".txt", 'rb') as file:
            return pickle.load(file)
    if method == 'RSGF full':
        with open('dump/'+filename+"_RSGF_logreg_full_noise_steps_const_"+str(args[0])+"_tun_steps_"+str(args[1])
                  +'_epochs_'+str(args[2])+"_delta_"+str(args[3])+".txt", 'rb') as file:
            return pickle.load(file)
    

def make_plots(args):
    supported_modes_y = ['func_vals']
    supported_modes_x = ['time', 'oracle_calls', 'iters']
    
    filename = args[0]
    mode_y = args[1]
    mode_x = args[2]
    figsize = args[3]
    sizes = args[4]
    title = args[5]
    methods = args[6]
    bbox_to_anchor = args[7]
    legend_loc = args[8]
    save_fig = args[9]
    
    
    title_size = sizes[0]
    linewidth = sizes[1]
    markersize = sizes[2]
    legend_size = sizes[3]
    xlabel_size = sizes[4]
    ylabel_size = sizes[5]
    xticks_size = sizes[6]
    yticks_size = sizes[7]
    
    assert(mode_y in supported_modes_y)
    assert(mode_x in supported_modes_x)
    
    fig = plt.figure(figsize=figsize)
    plt.title(title, fontsize=title_size)
    marker = itertools.cycle(('+', 'd', 'x', 'o', '^', 's', '*', 'p', '<', '>', '^'))
    
    num_of_methods = len(methods)
    for idx, method in enumerate(methods):
        res = read_results_from_file(filename, method[0], method[1])
        if method[3] == None:
            length = len(res['iters'])
        else:
            length = method[3]
        plt.semilogy(res[mode_x][0:length], res[mode_y][0:length] / res[mode_y][0], linewidth=linewidth, marker=next(marker), 
            markersize = markersize, 
            markevery=range(-idx*int(length/(10*num_of_methods)), len(res[mode_x][0:length]), int(length/10)), 
            label = method[2])
        
    
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=legend_loc, fontsize=legend_size)
    if mode_x == 'time':
        plt.xlabel(r"Time, $s$", fontsize=xlabel_size)
    if mode_x == 'oracle_calls':
        plt.xlabel(r"Number of oracle calls", fontsize=xlabel_size)
    if mode_x == 'iters':
        plt.xlabel(r"Number of iterations", fontsize=xlabel_size)
    if mode_y == 'func_vals':
        plt.ylabel(r"$\frac{f(x_k)-f(x^*)}{f(x_0)-f(x^*)}$", fontsize=ylabel_size)
    
    plt.xticks(fontsize=xticks_size)
    _ = plt.yticks(fontsize=yticks_size)
    
    ax = fig.gca()
    ax.xaxis.offsetText.set_fontsize(xticks_size - 2)
    ax.yaxis.offsetText.set_fontsize(yticks_size - 2)
    
    if save_fig[0]:
        plt.savefig("plot/"+save_fig[1], bbox_inches='tight')
