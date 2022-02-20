import os
import re
import sys
import time
import psutil
import argparse
import itertools
import numpy as np
from math import floor
from scipy import sparse
from random import random
from scipy.optimize import linprog
import scipy.special
import pandas as pd

################################################################################
# debug info
def debug_details(process, tstart, processed_matrix, vector_matrix, logfile):
    logfile.write("=======================================================================\n")
    logfile.write("RUN STATISTICS: FEASIBILITY\n")
    logfile.write("--------------\n")
    logfile.write("CPU:           [CORES-TOTAL: %s, THREADS-TOTAL: %s, THREADS-USED: %s]\n"%\
                     (psutil.cpu_count(logical=False), psutil.cpu_count(), process.num_threads()))
    logfile.write("\t       [CUR-FREQ: %f, MIN-FREQ: %f, MAX-FREQ: %f]\n"%\
                     (psutil.cpu_freq().current, psutil.cpu_freq().min,\
                      psutil.cpu_freq().max))
    logfile.write("MEMORY:        [RAM-MEM: %.2fMB, VIR-MEM: %.2fMB, TOTAL: %.2fMB]\n"%\
                     (process.memory_info().rss/(1024*1024),\
                      process.memory_info().vms/(1024*1024),\
                      (process.memory_info().rss+process.memory_info().vms)/(1024*1024)))
    logfile.write("MATRIX-MEMORY: [PROCESSED: %.2fMB, VECTOR: %.2fMB]\n"%\
                       (processed_matrix.nbytes/(1024*1024),\
                        vector_matrix.nbytes/(1024*1024)))
    logfile.write("TOTAL-TIME:    %.2fs\n"%(time.time()-tstart))
    logfile.write("======================================================================\n")
    logfile.flush()
#end debug_details()

################################################################################
# miscellaneous
def bin_to_int(bit_vector):
    int_value = 0
    for bit in bit_vector:
        int_value = (int_value << 1) | bit
    #end for
    return int_value
#end bin_to_int()

def int_to_bin(int_value, length):
    bit_vector= np.zeros(length, dtype=np.uint8)
    bit_index = 0
    while(int_value > 0):
        bit = int_value & 0x01
        bit_vector[bit_index] = bit
        int_value = int_value >> 1
        bit_index += 1
    #end while
    return bit_vector
#end int_to_bin()

def matrix_multiplication(A, B, logfile):
    tstart= time.time()
    C = A.dot(B)
    logfile.write("matrix-multiplication: [input-size: (%d, %d) X (%d, %d)]"%\
                     (A.shape[0], A.shape[1], B.shape[0], B.shape[1]))
    logfile.write(" [output-size: (%d, %d)]"%(C.shape[0], C.shape[1]))
    logfile.write(" [total-time: %.2fs]\n"%(time.time()-tstart))
    logfile.flush()
    return C
#end matrix_multiplication()

###############################################################################
# input processing
def get_color_matrix(color_matrix_file, logfile):
    tstart= time.time()
    if not os.path.exists(color_matrix_file):
        sys.stderr.write("File '%s' do not exist\n"%(color_matrix_file))
    #end if
    color_matrix = np.loadtxt(fname=color_matrix_file, delimiter=",", dtype=np.uint16)

    logfile.write("get-color-matrix: [num-colors: %d] [total-time: %.2fs]\n"%\
                     (color_matrix.shape[0], time.time()-tstart))
    logfile.flush()

    return color_matrix
#end get_distance_matrix()

def get_subset_map_and_size_list(color_matrix, logfile):
    tstart = time.time()
    t, N = color_matrix.shape
    for i in range(t):
        color_matrix[i] = color_matrix[i] * pow(2,t-i-1)
    #end for
    records_array = np.sum(color_matrix, axis=0)
    subset_idx, subset_size_list = np.unique(records_array, return_counts=True)
    subset_map = pd.DataFrame(records_array).groupby([0]).indices

    logfile.write("get-subset-map-and-size-list: [total-subsets: %s, num-subsets: %d]"%\
                   (pow(2,color_matrix.shape[0]), len(subset_map.keys())))
    logfile.write(' [total: %.2fs]\n'%(time.time()-tstart))
    logfile.flush()

    return subset_map, subset_size_list
#end get_subset_map_and_size_list()

def get_subset_index_to_id(subset_map):
    subset_index_to_id = {}
    index = 0
    for key in sorted(subset_map.keys()):
        subset_index_to_id[index] = key
        index += 1
    #end for
    return subset_index_to_id
#end  get_subset_index_to_id()

def get_processed_matrix(subset_map, t, use_sparse, logfile):
    tstart= time.time()
    processed_matrix = np.empty((0,t), np.int8)
    for key in sorted(subset_map.keys()):
        bit_vec = int_to_bin(key, t)
        processed_matrix = np.append(processed_matrix, [bit_vec], axis=0)
    #end for
    logfile.write("get-processed-matrix: [total: %.2fs]\n"%(time.time()-tstart))
    logfile.flush()

    if use_sparse:
        return sparse.csr_matrix(processed_matrix.transpose())
    return processed_matrix.transpose()
#end get_processed_matrix()

def generate_vector_matrix(subset_size_list, length, logfile):
    lattice_size = subset_size_list.size

    tstart= time.time()
    ## generate all combinations of vectors with repetitions 
    _list = range(lattice_size)
    combinations_with_replacement_size = int(scipy.special.binom(lattice_size + length - 1, length))
    comb_time = time.time() - tstart # record time for generating combinations
    
    vec_time_start = time.time()
    ## create a matrix with all vectors
    vector_matrix = np.zeros((combinations_with_replacement_size, lattice_size), dtype=np.uint8)
    for i, row in enumerate(itertools.combinations_with_replacement(_list, length)):
        np.add.at(vector_matrix[i], np.array(row), 1)
    #end for
    vec_time = time.time() - vec_time_start

    vec_proc_start = time.time()
    feasible_row_indices = np.where((vector_matrix <= subset_size_list).all(axis=1))[0]
    vector_matrix_processed = vector_matrix[feasible_row_indices, :]
    vec_proc_time = time.time() - vec_proc_start

    logfile.write("generate-vector-matrix: [vec-total: %d, vec-processed: %d]"%\
                     (vector_matrix.shape[0], vector_matrix_processed.shape[0]))
    logfile.write(" [comb-time: %.2fs, vec-time:%.2fs, vec-proc-time: %.2fs] [total: %.2fs]\n"%\
                      (comb_time, vec_time, vec_proc_time, time.time()-tstart))
    logfile.flush()

    return vector_matrix_processed.transpose()
#end generate_vector_matrix()

###############################################################################
# finding feasible solutions
def get_feasible_solutions(processed_matrix, vector_matrix, requirements, logfile):
    tstart= time.time()
    res_matrix = matrix_multiplication(processed_matrix, vector_matrix, logfile)
    matmul_time = time.time() - tstart

    feasible_row_indices = np.where((res_matrix.transpose() >= requirements).all(axis=1))[0]

    feasible_solutions = vector_matrix.transpose()[feasible_row_indices, :]
    logfile.write("get-feasible-solutions: [matrix-mul: %.2fs, total:%.2fs]"%\
                     (matmul_time, time.time()-tstart))
    logfile.write(" [total: %d, feasible: %d]\n"%\
                      (res_matrix.shape[1], feasible_solutions.shape[0]))
    logfile.flush()
    return feasible_solutions
#end get_feasible_solutions()

def get_any_feasible_solution(processed_matrix, requirements, subset_size_list, 
                              length, batch_size, logfile):
    lattice_size = subset_size_list.size
    tstart= time.time()
    _list = range(lattice_size)
    result = None
    iterator = itertools.combinations_with_replacement(_list, length)
    elements = []
    while(True):
        el = next(iterator, None)
        if el is not None:
            elements.append(list(el))
        #end if

        if (el is None or len(elements) == batch_size) and len(elements) != 0:
            comb_matrix = np.array(elements, dtype='uint8')
            rows, _ = comb_matrix.shape
            vector_matrix = np.zeros((rows, lattice_size), dtype=np.uint8)
            for row in range(rows):
                np.add.at(vector_matrix[row], elements[row],1)
            #end for
            elements = []
            feasible_row_indices = np.where((vector_matrix <= subset_size_list).all(axis=1))[0]
            vector_matrix_processed = vector_matrix[feasible_row_indices, :].transpose()
            res_matrix = processed_matrix.dot(vector_matrix_processed)
            feasible_row_indices = np.where((res_matrix.transpose() >= requirements).all(axis=1))[0]
            if len(feasible_row_indices) > 0:
                feasible_solution = vector_matrix_processed.transpose()[feasible_row_indices[0]]
                result = feasible_solution
                break;
            #end if
        if el is None:
            break;
        #end if
    #end while

    logfile.write("get-any-feasible-solution: [total:%.2fs]\n"% (time.time()-tstart))
    return result
#end get_any_feasible_solution()


###############################################################################
# linear program implementation
# The function works in two ways:
# 1) many_solution_lp_only = False
# returns a single feasible solution and make a few retries
# randomizng the objecttive function and reruning randomized raouding 
# 2) many_solution_lp_only = True
# Here the program does not stop after the first success
# but returns as many solutions as possible within limited retrials
# This is used by LP_LS heuristic.
def get_linear_programming_solution(subset_size_list,\
                                    processed_matrix,\
                                    requirement_list,\
                                    num_of_medians,\
                                    many_solution_lp_only = False):
    v = np.ones(subset_size_list.size, dtype=np.uint8)
    if many_solution_lp_only:
        A = np.multiply(processed_matrix, -1)
        B = np.multiply(requirement_list, -1)
    else:
        A = np.append(np.multiply(processed_matrix, -1), [v], axis=0)
        B = np.append(np.multiply(requirement_list, -1), num_of_medians)   
    number_of_retries_1 = 50 if many_solution_lp_only else 5
    number_of_retries_2 = 2 if many_solution_lp_only else 3
    many_solutions = []
    bounds = list(map(lambda x: (0, x), subset_size_list))
    for _ in range(number_of_retries_1):
        f = np.random.rand(subset_size_list.size)
        if many_solution_lp_only:
            result = linprog(f,A,B, A_eq = [v], b_eq = [num_of_medians] ,bounds=bounds)
        else:
            result = linprog(f,A,B, bounds=bounds)
        if not result.success:
            break
        #end if
        for __ in range(number_of_retries_2):
            rounded_result = np.array(list(map(lambda x: floor(x + random()), result.x)))
            if (rounded_result <= subset_size_list).all(axis=0):
                res_matrix = processed_matrix.dot(rounded_result.reshape(-1, 1))
                if (res_matrix.transpose() >= requirement_list).all(axis=1)[0]:
                    if many_solution_lp_only:
                        many_solutions.append(rounded_result)
                    else:
                        return rounded_result
                #end if
            #end if
        #end for
    #end for
    return many_solutions if many_solution_lp_only else None
#end get_linear_programming_solution()


###############################################################################
# dynamic programming algorithm

def int_to_k_array(a, num_of_medians, num_colors):
    result = np.zeros(num_colors, dtype=np.int8)
    i = num_colors - 1
    while (a != 0):
        result[i] = a % (num_of_medians + 1)
        a = floor(a / (num_of_medians + 1))
        i = i - 1
    #end while
    return result
#end int_to_k_array()
    
def k_array_to_int(a, num_of_medians):
    m = 1
    result = 0
    for i in range (a.size, 0, -1):
        result = result + a[i - 1] * m
        m = m * (num_of_medians + 1)
    return result
#end k_array_to_int()

def get_dynamic_programming_solution(processed_matrix, 
                                     num_colors, 
                                     num_of_medians, 
                                     subset_size_list, 
                                     requirement_list,
                                     logfile):
    tstart = time.time()
    for r in requirement_list:
        if r > num_of_medians:
            return None
        #end if
    #end for

    # preprocessing
    time_buf = time.time()
    converted_subset_size_list = list(map(lambda subset: min(num_of_medians, subset), subset_size_list))
    E_ = np.zeros((sum(converted_subset_size_list), num_colors), dtype=np.int8)
    processed_matrix_transposed = processed_matrix.transpose()
    i = 0
    for idx, size in enumerate(converted_subset_size_list):
        for _ in range(size):
            E_[i] = processed_matrix_transposed[idx]
            i=i+1
        #end for
    #end for

    # allocate space for the dynamic programming array
    nof_rows = sum(converted_subset_size_list) + 1
    nof_colums = pow(num_of_medians + 1, num_colors)
    A = np.empty((nof_rows, nof_colums), dtype=np.int8)
    B = np.negative(np.ones((2, nof_colums, num_of_medians), dtype=np.int32))
    
    preproc_time = time.time() - time_buf
    time_buf = time.time()
    # dynamic programming recursion
    A[0].fill(num_of_medians + 1)
    A[0, 0] = 0
    # colums greater than the req to be stripped
    allowed_columns = np.zeros(nof_colums, dtype=bool)
    for j in range(nof_colums):
        current_set = int_to_k_array(j, num_of_medians, num_colors)
        diff = requirement_list - current_set
        allowed_columns[j] = np.all((diff >= 0))
    #end for

    for i in range(1, nof_rows):
        for j in range(nof_colums):
            if not allowed_columns[j]:
                continue
            #end if
            first_candidate = A[(i - 1) % 2, j]
            current_set = int_to_k_array(j, num_of_medians, num_colors)
            previous_set = current_set - E_[i-1]

            previous_set[previous_set < 0] = 0
            previous_set_id = k_array_to_int(previous_set, num_of_medians)
            second_candidate = A[(i - 1) % 2, previous_set_id]
            new_value = min(first_candidate, second_candidate + 1, num_of_medians + 1)
            if new_value != A[i % 2, j]:
                A[i % 2, j] = new_value
                if new_value == first_candidate:
                    B[i % 2, j, :] = B[(i - 1) % 2, j, :]
                else:
                    B[i % 2, j, :] = B[(i - 1) % 2, previous_set_id, :]
                    for l in range(num_of_medians):
                        if B[i % 2, j, l] == -1:
                            B[i % 2, j, l] = i
                            break

        #end for
    #end for
    dynamic_time = time.time() - time_buf
    time_buf = time.time()

    req_id = k_array_to_int(requirement_list, num_of_medians)
    min_k = A[(nof_rows - 1) % 2, req_id]
    result = None
    if min_k > num_of_medians:
        return None
    #end if

    # get one solution
    #result = np.empty((min_k, num_colors), dtype=np.int8)
    result = list(map(lambda e: list(E_[e - 1]), B[(nof_rows - 1) % 2, req_id]))

    #end while
    result_time = time.time() - time_buf

    logfile.write("dynamic-program: [preproc: %.2fs, dynamic: %.2fs, result: %.2fs]"%\
                  (preproc_time, dynamic_time, result_time))
    total_time = time.time() - tstart
    logfile.write(" [total-time: %.2fs]\n"%(total_time))
    logfile.flush()

    return result
#end get_dynamic_programming_solution()

def feasible_solution_to_facilities(feasible_solution, subset_map):
    subset_index_to_id = get_subset_index_to_id(subset_map)
    facilities = []
    for idx, e in enumerate(feasible_solution):
        candidates = subset_map[subset_index_to_id[idx]]
        for i in range(e):
            facilities.append(candidates[i])
        #end for
    #end for
    return facilities
#end feasible_solution_to_facilities()

def many_feasible_solutions_to_facilitiess(feasible_solutions, subset_map):
    subset_index_to_id = get_subset_index_to_id(subset_map)
    result = []
    for feasible_solution in feasible_solutions:
        for _ in range(10):
            facilities = []
            for idx, e in enumerate(feasible_solution):
                candidates = np.copy(subset_map[subset_index_to_id[idx]])
                for _ in range(e):
                    random_index = np.random.randint(0, len(candidates))
                    facilities.append(candidates[random_index])
                    np.delete(candidates, random_index)
            result.append(facilities)
    result.append(facilities)
    return result
#end feasible_solution_to_facilities()

###############################################################################
def calculate(num_of_medians,\
              requirement_list,\
              color_matrix,\
              command,\
              return_solution = False,\
              logfile = sys.stdout,
              many_solution_lp_only = False):
    tstart  = time.time()
    process = psutil.Process()
    tstart_temp = time.time()
    
    ## generate subset map
    num_colors = len(requirement_list)
    subset_map, subset_size_list = get_subset_map_and_size_list(color_matrix,\
                                                            logfile)

    input_time  = time.time() - tstart_temp
    tstart_temp = time.time()

    ## generate matrix based on subset lattice
    use_sparse_matrix = False
    processed_matrix = get_processed_matrix(subset_map, num_colors,
                                            use_sparse_matrix, logfile)
    processed_time   = time.time() - tstart_temp
    tstart_temp      = time.time()
    vector_matrix    = np.zeros(0)
    solution = None

    vector_time = 0.0
    if command == 'linear-program':
        tstart_temp = time.time()
        feasible_solution = get_linear_programming_solution(
                                    subset_size_list, 
                                    processed_matrix, 
                                    requirement_list, 
                                    num_of_medians,
                                    many_solution_lp_only=many_solution_lp_only)
        nof_solutions = 0 if feasible_solution is None else 1
        if many_solution_lp_only:
            solution = feasible_solution
        if return_solution:
            if nof_solutions:
                solution = feasible_solution_to_facilities(feasible_solution, subset_map)
            else:
                solution = []
            #end if
        #end if
    elif command == 'dynamic-program':
        tstart_temp = time.time()
        # get a feasible solution
        feasible_solution = get_dynamic_programming_solution(
                                    processed_matrix, 
                                    num_colors, 
                                    num_of_medians, 
                                    subset_size_list, 
                                    requirement_list,
                                    logfile)
        nof_solutions = 0 if feasible_solution is None else 1
        if return_solution:
            facilities = []
            for row in feasible_solution:
                group_id = bin_to_int(np.flip(row))
                candidates = subset_map[group_id]
                facilities.append(candidates[0])
            #end for
            solution = facilities
        #end if
    elif command == 'brute-force':
        batch_size = 100
        tstart_temp = time.time()
        feasible_solution = get_any_feasible_solution(processed_matrix,\
                                                    requirement_list,\
                                                    subset_size_list,\
                                                    num_of_medians,\
                                                    batch_size,\
                                                    logfile)    
        nof_solutions = 0 if feasible_solution is None else 1
        if return_solution:
            solution =  feasible_solution_to_facilities(feasible_solution, subset_map)
        #end if
    else:
        ## generate vectors using intersection subset lattice
        vector_matrix = generate_vector_matrix(subset_size_list,\
                                               num_of_medians,\
                                               logfile)
        vector_time   = time.time() - tstart_temp
        tstart_temp   = time.time()
        feasible_solutions = get_feasible_solutions(processed_matrix,\
                                                    vector_matrix,\
                                                    requirement_list,\
                                                    logfile)
        solution = feasible_solutions
        nof_solutions = feasible_solutions.shape[0]
    #end if

    feasible_time = time.time() - tstart_temp
    ## log runtime
    logfile.write("calculate: [input: %.2fs, processed: %.2fs, vector: %.2fs, feasible: %.2fs]"%\
                     (input_time, processed_time, vector_time, feasible_time))
    total_time = time.time()-tstart
    logfile.write(" [total-time: %.2fs]\n"%(total_time))

    ## log debug information
    debug_details(process, tstart, processed_matrix, vector_matrix, logfile)

    #update performance statistics
    perf_stats = {}
    perf_stats['input_time'] = input_time
    perf_stats['solution'] = solution
    perf_stats['subset_map'] = subset_map
    perf_stats['processed_time'] = processed_time
    perf_stats['vector_time'] = vector_time
    perf_stats['feasible_time'] = feasible_time
    perf_stats['total_time'] = total_time
    perf_stats['nof_solutions'] = nof_solutions
    perf_stats['proc_matrix_shape'] = processed_matrix.shape
    perf_stats['vector_matrix_shape'] = vector_matrix.shape
    perf_stats['peak_memory'] = process.memory_info().rss/(1024*1024)
    perf_stats['virtual_memory'] = process.memory_info().vms/(1024*1024)

    return perf_stats
    #return (input_time, processed_time, vector_time, feasible_time,
    #        total_time, nof_solutions, processed_matrix[0].size)
#end calculate()

###############################################################################
## command line arguments parser
def cmd_parser():
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group('Arguments')
    #g.add_argument('-distances', '--distance-matrix', nargs='?', required=False,\
    #               type=str, default='distances.csv')
    g.add_argument('-colors', '--color-matrix', nargs='?', required=False,\
                   type=str, default='colors.csv')
    g.add_argument('-k', '--num-of-medians', nargs='?', required=True, type=int,\
                   default=2)
    g.add_argument('-rvec', '--requirements', nargs='+',\
                   required=True, type=int, default=[1,1])
    g.add_argument('-cmd', '--command', nargs='?', required=True, type=str,\
                   default='all', choices=['all', 'brute-force', 'dynamic-program', 'linear-program'])
    g.add_argument('-log', '--log-file', nargs='?', required=False, type=str,\
                   default='stdout')
    return parser
#end cmd_parser()


###############################################################################
## main function
def main():
    tstart = time.time()

    ## handling command line arguments
    parser = cmd_parser()
    opts   = vars(parser.parse_args())
    #dist_matrix_file  = opts['distance_matrix']
    color_matrix_file   = opts['color_matrix']
    num_of_medians      = opts['num_of_medians']
    requirement_list    = np.array(opts['requirements'], dtype='uint8')
    command             = opts['command']
    output_filename        = opts['log_file']

    if output_filename == 'stdout':
        logfile = sys.stdout
    else:
        logfile = open(output_filename, 'a')
    #end if
    logfile.write('\n\n')
    logfile.flush()

    # read color matrix
    color_matrix = get_color_matrix(color_matrix_file, logfile)

    # find a feasible solution
    calculate(num_of_medians, requirement_list, color_matrix, command)

#end main()

## program entry point
if __name__=="__main__":
    main()
