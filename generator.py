import sys
import math
import random
import numpy as np
from feasibility import calculate
from sklearn.datasets import make_blobs


###############################################################################
def generate_groups(nof_groups = 5,\
                    nof_facilities = 10,\
                    max_freq = 3,\
                    seed=123456789):
    #error handling
    if(max_freq > nof_groups):
        sys.stderr.write("invalied parameter values: mu = %d, t = %d\n"%\
                          (max_freq, nof_groups))
    #end if
    
    random.seed(seed)
    result = np.zeros((nof_groups,nof_facilities), dtype=np.int8)
    for j in range (0, nof_facilities):
        freq = min(math.ceil(random.random() * max_freq), nof_groups)
        groups = random.sample(range(nof_groups), freq)
        for i in groups:
            result[i, j] = 1
        #end for
    #end for
    return result
#end generate_groups()

###############################################################################
def generate_requirements(nof_groups = 5,\
                          r_max = 3,\
                          r_min = 1,\
                          seed=123456789):
    #error handling
    if(r_max > nof_groups or r_min < 1):
        sys.stderr.write("invalid parameter values: r_max = %d, r_min = %d, t = %d"%\
                         (r_max, r_min, nof_groups))
    #end if

    random.seed(seed)
    result = []
    for i in range (0, nof_groups):
        # square distribution
        a = r_min + math.floor((1 - math.sqrt(random.random())) * (r_max - r_min + 1))
        result.append(a)
    #end for
    return np.array(result)
#end generate_requirements()

###############################################################################
def create_unique_solution(rvec, groups, k):
    nof_facilities = len(groups[0])
    nof_groups = len(groups)
    
    #require 'k' facilities from group 't'
    rvec[len(rvec) - 1] = k
    
    for i in range (0, nof_groups):
        for j in range (nof_facilities - k, nof_facilities):
            groups[i][j] = 1
        #end for
        for j in range (0, nof_facilities - k):
            groups[0][j] = 0
        #end for
    #end for
    
    solution = []
    for j in range(nof_facilities - k, nof_facilities):
        solution.append(j+1)
    #end for
    
    return rvec, groups, np.array(solution)
#end create_unique_solution()

###############################################################################
def get_feasible_instance(nof_groups,\
                         nof_facilities,\
                         r_max,\
                         r_min,\
                         max_freq,\
                         k,\
                         seed,\
                         unique=False):
    #error handling
    if(r_max > k or r_min < 1):
        sys.stderr.write("invalid parameter value: r_max = %d, r_min = %d, k = %d\n"%\
                         (r_max, r_min, k))
    #end if
    if(max_freq > nof_groups):
        sys.stderr.write("invalid parameter value: mu = %d, t = %d\n"%\
                         (max_freq, nof_groups))
    #end if
    
    #initiliaze random number generator
    random.seed(seed)
    rvec_seed   = random.randint(1, sys.maxsize)
    groups_seed = random.randint(1, sys.maxsize)
    
    rvec   = generate_requirements(nof_groups, r_max, r_min, rvec_seed)
    groups = generate_groups(nof_groups, nof_facilities, max_freq, groups_seed)
    
    solution = np.zeros(k)
    if(unique):
        rvec, groups, solution = create_unique_solution(rvec, groups, k)
    #end if
    
    return groups, rvec, solution
#end get_feasible_instance()

###############################################################################
# generate distance matrix
def is_symmetric_matrix(A):
    tol = 1e-8
    return np.all(np.abs(A - A.T) < tol)
#end is_symmetric_matrix()

def get_distance_matrix(n, seed=123456789):
    np.random.seed(seed)
    # matrix generated from uniform distribution
    # NOTE: distance function is metric
    A = np.random.uniform(low=0.5, high=1.0, size=(n, n))
    A_symm = np.tril(A) + np.tril(A, -1).T
    np.fill_diagonal(A_symm, 0.0)
    return A_symm 
#end generate_random_matrix()

###############################################################################
def get_dataset_instance(N, d, k, seed=0, c_std=0.8):
    data, _ = make_blobs(n_samples=N,
                         centers=k,
                         n_features=d,
                         random_state=seed,
                         cluster_std=c_std)
    return data
#end get_dataset_instance()
