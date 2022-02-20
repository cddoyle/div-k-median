#python imports
import re
import sys
import time
import random
import argparse
import itertools
from re import S
import numpy as np
import pandas as pd
from pprint import pprint
from subprocess import call
from datetime import datetime
from sklearn.datasets import make_blobs
from numpy.lib.arraysetops import unique


#local imports
import generator
import feasibility
import local_search
from local_search import kmedian_local_search
from kmeans import kmedoids_sklearn

def parse_result(result):
    return result.astype({'n':'int32',\
                          't':'int32',\
                          'k':'int32',\
                          'r_min':'int32',\
                          'r_max':'int32',\
                          'max_freq':'int32',\
                          'seed':'int64',\
                          'instance_time':'float64',\
                          'feasibility_time':'float64',\
                          'objective_time':'float64'})
#end parse_result()




################################################################################
def scalability(nof_facilities,\
                nof_groups,\
                nof_centers,\
                nof_iterations,\
                given_r_max,\
                unique,\
                command,\
                logfile,\
                result_file,\
                objective,
                many_solutions_lp_only = False):
    #logfile.write("%6s %3s %3s %5s %5s %8s %19s %9s\n"%\
    #              ("n", "t", "k", "r_min", "r_max",
    #               "max_freq", "seed", "inst_time"))
    #logfile.write("-------------------------------------------------------------\n")
    stats = pd.DataFrame(columns=['n', 't', 'k', 'r_min', 'r_max', 'max_freq',\
                                  'seed', 'instance_time', 'feasibility_time',\
                                  'virtual_memory', 'peak_memory',\
                                  'objective_time', 'input_time_', 'processed_time_', 'vector_time_', 'feasible_time_', 'total_time_'],\
                                  dtype='int32')
    for n, t, k in itertools.product(nof_facilities, nof_groups, nof_centers):
        for i in range(nof_iterations):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print(n, t, k, i, command)

            #random number generator seeds
            gen_seed  = random.randint(1, sys.maxsize)
            dist_matrix_seed = random.randint(1, int(pow(2, 32)-1))
            local_search_seed = random.randint(1, int(pow(2, 32)-1))

            #initialize
            r_max = given_r_max or min(t, k)
            r_min = 1
            max_freq = int(t/2)+1

            #generate instance and time it
            time_buf = time.time()
            color_mat, rvec, solution = generator.get_feasible_instance(
                                              t,
                                              n,
                                              r_max,
                                              r_min,
                                              max_freq,
                                              k,
                                              gen_seed,
                                              unique)
            instance_time = time.time() - time_buf
            
            #find a feasible solution and time it
            time_buf = time.time()
            return_solution = False
            perf_stats = feasibility.calculate(k, rvec, color_mat, 
                                               command, return_solution,
                                               logfile, many_solutions_lp_only)
            feasibility_time = time.time() - time_buf

            #find cluster centers based on objective and time it
            objective_time = 0
            dist_matrix_time = 0
            ls_stats = {}
            if objective != None:
                time_buf = time.time()
                dist_matrix = generator.get_distance_matrix(n, dist_matrix_seed)
                dist_matrix_time = time.time() - time_buf

                time_buf = time.time()
                ls_stats = kmedian_local_search(dist_matrix, 
                                                     k,
                                                     local_search_seed,
                                                     0.0)
                objective_time = time.time() - time_buf
            #end if

            #printing stats
            pprint(perf_stats, stream=logfile)
            pprint(ls_stats, stream=logfile)

            peak_memory    = perf_stats['peak_memory']
            virtual_memory = perf_stats['virtual_memory']
            virtual_memory = perf_stats['virtual_memory']
            input_time_ = perf_stats['input_time']
            processed_time_ = perf_stats['processed_time']
            vector_time_ = perf_stats['vector_time']
            feasible_time_ = perf_stats['feasible_time']
            total_time_ = perf_stats['total_time']
            logfile.write("%6d %3d %3d %5d %5d %8d %d"%\
                          (n, t, k, r_min, r_max, max_freq, gen_seed))
            logfile.write(" %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f"%\
                          (instance_time, feasibility_time, dist_matrix_time, objective_time, input_time_, processed_time_, vector_time_, feasible_time_, total_time_))
            logfile.write("\n======================================================================\n")
            logfile.write("\n\n\n")
            logfile.flush()

            #append results to pandas dataframe
            stats.loc[len(stats)] = [n, t, k, r_min, r_max, max_freq,\
                                     gen_seed, instance_time, feasibility_time,\
                                     virtual_memory, peak_memory, objective_time, \
                                     input_time_, processed_time_, vector_time_, feasible_time_, total_time_]
            result_file.seek(0)
            result_file.truncate()
            #change datatype of columns
            result = parse_result(stats)
            result_file.write(result.to_string())
            result_file.write("\n----\n")
            result_file.write(result.to_json(orient='records'))
            result_file.flush()
        #end for
    #end for

    #change datatype of columns
    return parse_result(stats)
#end scalability()



###############################################################################
def scaling_nof_facilities(command,\
                           unique = True,\
                           range = [100, 1000, 10000, 100000, 1000000, 10000000],\
                           objective = None,\
                           results_dir = 'exp-results',\
                           test_run = False,
                           many_solutions_lp_only = True):
    print("scaling_nof_facilities", command, unique, range)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities  = range
    #nof_facilities  = list(range(1000, 20001, 1000))
    nof_groups      = [7]
    nof_centers     = [4]
    nof_iterations  = 10
    r_max           = None
    return_solution = False

    logfile_name = "./%s/scaling_nof_facilities-%s.log"%\
                    (results_dir, command)
    logfile = open(logfile_name, 'w')
    #if test_run:
    #    logfile = sys.stdout

    result_file_name = "./%s/scaling_nof_facilities-%s%s.results"%\
                       (results_dir, command, '-unique' if unique else '')
    result_file = open(result_file_name, 'w')

    stats = scalability(nof_facilities,\
                        nof_groups,\
                        nof_centers,\
                        nof_iterations,\
                        r_max,\
                        unique,\
                        command,\
                        logfile,\
                        result_file,\
                        objective,
                        many_solutions_lp_only)
    pprint(stats)
#end scaling_nof_facilities()

def scaling_nof_centers(command,\
              unique = True,\
              range = [4, 5, 6, 7, 8, 9],\
              objective = None,\
              results_dir = 'exp-results',\
              test_run = False,
              many_solutions_lp_only = True):
    print("scaling_nof_centers", command, unique, range)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities  = [10000]
    nof_groups      = [6]
    # 320 sec with 8 in the worst case with BF, expect 1600 sec with 9, so the total time about 5 hours
    nof_centers     = range
    nof_iterations  = 10
    r_max           = 3
    return_solution = False

    logfile_name = "./%s/scaling_nof_centers-%s%s.log"%\
                   (results_dir, command, '-unique' if unique else '')
    logfile = open(logfile_name, 'w')
    #if test_run:
    #    logfile = sys.stdout

    result_file_name = "./%s/scaling_nof_centers-%s%s.results"%\
                       (results_dir, command, '-unique' if unique else '')
    result_file = open(result_file_name, 'w')

    stats = scalability(nof_facilities,\
                        nof_groups,\
                        nof_centers,\
                        nof_iterations,\
                        r_max,\
                        unique,\
                        command,\
                        logfile,\
                        result_file,\
                        objective,
                        many_solutions_lp_only)
    
    pprint(stats)
#end scaling_nof_centers()

def scaling_nof_groups(command,\
                       unique = True,\
                       range = [4, 5, 6, 7, 8],\
                       objective = None,\
                       results_dir = 'exp-results',
                       test_run = False,
                       many_solutions_lp_only = True):
    print("scaling_nof_groups", command, unique, range)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities  = [10000]
    # 1200 sec with 8 in the worst case, expect 4 hours in total
    #nof_groups      = [4, 5, 6, 7, 8]
    nof_groups      = range
    nof_centers     = [5]
    nof_iterations  = 10
    r_max           = 3
    return_solution = False

    logfile_name = "./%s/scaling_nof_groups-%s%s.log"%\
                   (results_dir, command, '-unique' if unique else '')
    logfile = open(logfile_name, 'w')
    #if test_run:
    #    logfile = sys.stdout

    result_file_name = "./%s/scaling_nof_groups-%s%s.results"%\
                       (results_dir, command, '-unique' if unique else '')
    result_file = open(result_file_name, 'w')

    stats = scalability(nof_facilities,\
                        nof_groups,\
                        nof_centers,\
                        nof_iterations,\
                        r_max,\
                        unique,\
                        command,\
                        logfile,\
                        result_file,\
                        objective,
                        many_solutions_lp_only)
    pprint(stats)
#end scaling_nof_groups()

###############################################################################
def test_batch_scaling(objective, results_dir, test_run = False):
    for unique in [True, False]: # worst case first
        for algo_type in ['linear-program', 'brute-force', 'dynamic-program']:
            ################################
            # scaling 'k'
            if test_run:
                range_data = list(range(4,5))
            elif algo_type == 'linear-program':
                range_data = list(range(4, 30))
            elif algo_type == 'brute-force':
                if unique:
                    range_data = list(range(4, 10))
                else:
                    range_data = list(range(4, 30))
                #end if
            elif algo_type == 'dynamic-program':
                if unique:
                    range_data = list(range(4, 13))
                else:
                    range_data = list(range(4, 11))
                #end if
            #end if

            scaling_nof_centers(algo_type,\
                                unique,\
                                range_data,\
                                objective,\
                                results_dir,\
                                test_run)

            ################################
            #scaling 't'
            if test_run:
                range_data = list(range(4, 5))
            elif algo_type == 'linear-program':
                range_data = list(range(4, 15))
            elif algo_type == 'brute-force':
                if unique:
                    range_data = list(range(4, 9)) # 9 takes over 2 hours, predict 40 for tests
                else:
                    range_data = list(range(4, 12))
            elif algo_type == 'dynamic-program':
                range_data = list(range(4, 9))
            #end if
            scaling_nof_groups(algo_type,\
                               unique,\
                               range_data,\
                               objective,\
                               results_dir,\
                               test_run)

            ################################
            # scaling 'n'
            if test_run:
                range_data = [100]
            else:
                range_data = np.logspace(3,9,num=9-3, endpoint=False).astype(int) # use 10 in the final version
            #end if
            # override to test script
            scaling_nof_facilities(algo_type,\
                                   unique,\
                                   range_data,\
                                   objective,\
                                   results_dir,\
                                   test_run)
        #end for
    #end for
#end test_batch_scaling()


def test_batch_scaling_LP_search(objective, results_dir, test_run = False):
    algo_type = 'linear-program'
    unique = False
    ################################
    # scaling 'k'
    range_data = list(range(4, 7))
    #end if

    scaling_nof_centers(algo_type,\
                        unique,\
                        range_data,\
                        objective,\
                        results_dir,\
                        test_run,\
                        many_solutions_lp_only = True)

    ################################
    #scaling 't'
    range_data = list(range(4, 8))
    scaling_nof_groups(algo_type,\
                        unique,\
                        range_data,\
                        objective,\
                        results_dir,\
                        test_run,\
                        many_solutions_lp_only = True)

    ################################
    # scaling 'n'
    range_data = np.logspace(3,6,num=6-3, endpoint=False).astype(int) # use 10 in the final version
    #end if
    # override to test script
    scaling_nof_facilities(algo_type,\
                            unique,\
                            range_data,\
                            objective,\
                            results_dir,\
                            test_run,\
                            many_solutions_lp_only = True)
        #end for
    #end for
#end test_batch_scaling()

def test_batch_scaling_feasibility():
    scaling_k('linear-program', False, list(range(4, 30)))
    scaling_k('brute-force', False, list(range(4, 30)))
    scaling_k('dynamic-program', False, list(range(4, 11)))
    scaling_k('linear-program', True, list(range(4, 30)))
    scaling_k('brute-force', True, list(range(4, 10)))
    scaling_k('dynamic-program', True, list(range(4, 13)))
    scaling_nof_groups('linear-program', False, list(range(4, 15)))
    scaling_nof_groups('brute-force', False, list(range(4, 12)))
    scaling_nof_groups('dynamic-program', False, list(range(4, 9)))
    scaling_nof_groups('linear-program', True, list(range(4, 15)))
    scaling_nof_groups('brute-force', True, list(range(4, 12)))
    scaling_nof_groups('dynamic-program', True, list(range(4, 9)))
    scaling_nof_facilities('linear-program', False, list(np.logspace(3,10,num=10-3, endpoint=False)))
    scaling_nof_facilities('brute-force', False, list(np.logspace(3,10,num=10-3, endpoint=False)))
    scaling_nof_facilities('dynamic-program', False, list(np.logspace(3,10,num=10-3, endpoint=False)))
    scaling_nof_facilities('linear-program', True, list(np.logspace(3,10,num=10-3, endpoint=False)))
    scaling_nof_facilities('brute-force', True, list(np.logspace(3,10,num=10-3, endpoint=False)))
    scaling_nof_facilities('dynamic-program', True, list(np.logspace(3,10,num=10-3, endpoint=False)))
#end test_batch_scaling_feasibility()

################################################################################
# scaling local search

def parse_stats_LS(result):
    return result.astype({'n':'int32',\
                          'k':'int32',\
                          'swaps':'int32',\
                          'dist_matrix_seed':'int64',\
                          'local_search_seed':'int64',\
                          'dist_matrix_time':'float64',\
                          'objective_time':'float64',\
                          'virtual_memory':'float64',\
                          'peak_memory':'float64'})
#end parse_result()


d = 2
def scalability_optimal(nof_facilities,\
                            nof_centers,\
                            nof_swaps,\
                            nof_iterations,\
                            objective,\
                            logfile,\
                            result_file,\
                            strategy='local_search_2'):
    #logfile.write("%6s %3s %3s %5s %5s %8s %19s %9s\n"%\
    #              ("n", "t", "k", "r_min", "r_max",
    #               "max_freq", "seed", "inst_time"))
    #logfile.write("-------------------------------------------------------------\n")
    stats = pd.DataFrame(columns=['n', 'k', 'swaps', 'strategy', 'objective',\
                                  'dist_matrix_seed', 'local_search_seed',\
                                  'dist_matrix_time', 'objective_time',\
                                  'virtual_memory', 'peak_memory'],\
                                  dtype='int32')
    for n, k, swaps in itertools.product(nof_facilities, nof_centers, nof_swaps if strategy == 'local_search' else [0]):
        for i in range(nof_iterations):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print(n, k, i, objective, strategy)

            #random number generator seeds
            dist_matrix_seed = random.randint(1, int(pow(2, 32)-1))
            local_search_seed = random.randint(1, int(pow(2, 32)-1))

            #find cluster centers based on objective and time it
            computation_stats = {}
            time_buf = time.time()
            if strategy == 'local_search':
                dist_matrix = generator.get_distance_matrix(n, dist_matrix_seed)
                dist_matrix_time = time.time() - time_buf

                time_buf = time.time()
                computation_stats = local_search.local_search(dist_matrix, 
                                                    dist_matrix, 
                                                    k,
                                                    objective,
                                                    local_search_seed,
                                                    swaps,
                                                    logfile)
            elif strategy == 'kmeans_mlpack':
                data = np.random.uniform(low=0.0, high=1.0, size=(n, d))
                dist_matrix_time = time.time() - time_buf
                time_buf = time.time()
                computation_stats = kmeans_mlpack(data, k)
            elif strategy == 'kmeans_sklearn':
                data = np.random.uniform(low=0.0, high=1.0, size=(n, d))
                dist_matrix_time = time.time() - time_buf
                time_buf = time.time()
                computation_stats = kmeans_sklearn(data, k)
            elif strategy == 'local_search_2':
                data, _ = make_blobs(n_samples=n,
                    centers=k,
                    n_features=d,
                    random_state=0,
                    cluster_std=0.8)
                dist_matrix_time = time.time() - time_buf
                time_buf = time.time()
                computation_stats = kmedian_local_search(data, k)
            elif strategy == 'kmedoid':
                data = np.random.uniform(low=0.0, high=1.0, size=(n, d))
                dist_matrix_time = time.time() - time_buf
                time_buf = time.time()
                computation_stats = kmedoids_sklearn(data, k)
            #end if
            objective_time = time.time() - time_buf

            #printing stats
            pprint(computation_stats, stream=logfile)

            peak_memory    = computation_stats['peak_memory']
            virtual_memory = computation_stats['virtual_memory']
            logfile.write("%6d %3d %3d %d %d"%\
                          (n, k, swaps, dist_matrix_seed, local_search_seed))
            logfile.write(" %9.4f %9.4f %.2f %.2f"%\
                          (dist_matrix_time, objective_time, virtual_memory, peak_memory))
            logfile.write("\n======================================================================\n")
            logfile.write("\n\n\n")
            logfile.flush()

            #append results to pandas dataframe
            stats.loc[len(stats)] = [n, k, swaps, strategy, objective,\
                                     dist_matrix_seed, local_search_seed,\
                                     dist_matrix_time, objective_time,\
                                     virtual_memory, peak_memory]
            result_file.seek(0)
            result_file.truncate()
            #change datatype of columns
            result = parse_stats_LS(stats)
            result_file.write(result.to_string())
            result_file.write("\n----\n")
            result_file.write(result.to_json(orient='records'))
        #end for
    #end for

    #change datatype of columns
    return parse_stats_LS(stats)
#end scalability()

def scaling_nof_facilities_optimal(range_data,\
                           objective,\
                           results_dir,\
                           strategy = 'local_search',\
                           test_run = False):
    print("scaling_nof_facilities_optimal", range_data, strategy)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities  = range_data
    #nof_facilities  = list(range(1000, 20001, 1000))
    nof_centers     = [3]
    nof_swaps       = [1]
    nof_iterations  = 2 if test_run else 5
    r_max           = None
    return_solution = False

    logfile_name = "./%s/scaling_nof_facilities-%s.log"%\
                    (results_dir, strategy)
    logfile = open(logfile_name, 'w')

    result_file_name = "./%s/scaling_nof_facilities-%s.results"%\
                       (results_dir, strategy)
    result_file = open(result_file_name, 'w')

    stats = scalability_optimal(nof_facilities,\
                        nof_centers,\
                        nof_swaps,\
                        nof_iterations,\
                        objective,\
                        logfile,\
                        result_file,\
                        strategy)
    pprint(stats)
#end scaling_nof_facilities()

def scaling_nof_centers_optimal(range_data,\
                           objective,\
                           results_dir,\
                           strategy = 'local_search',\
                           test_run = False):
    print("scaling_nof_centers_optimal", range_data, strategy)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities  = [10000]
    nof_centers     = range_data
    nof_swaps       = [1]
    nof_iterations  = 2 if test_run else 5
    return_solution = False

    logfile_name = "./%s/scaling_nof_centers-%s.log"%\
                    (results_dir, strategy)
    logfile = open(logfile_name, 'w')

    result_file_name = "./%s/scaling_nof_centers-%s.results"%\
                       (results_dir, strategy)
    result_file = open(result_file_name, 'w')

    stats = scalability_optimal(nof_facilities,\
                        nof_centers,\
                        nof_swaps,\
                        nof_iterations,\
                        objective,\
                        logfile,\
                        result_file,\
                        strategy)
    pprint(stats)
#end scaling_nof_facilities()

def LS_scaling_nof_swaps(range_data,\
                           objective,\
                           results_dir,\
                           test_run = False):
    print("scaling_nof_swaps", range_data)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities  = [1000]
    nof_centers     = [4]
    nof_swaps       = range_data
    nof_iterations  = 5
    return_solution = False

    logfile_name = "./%s/scaling_nof_swaps-%s.log"%\
                    (results_dir, objective)
    logfile = open(logfile_name, 'w')
    if test_run:
       logfile = sys.stdout

    result_file_name = "./%s/scaling_nof_swaps-%s.results"%\
                       (results_dir, objective)
    result_file = open(result_file_name, 'w')

    stats = scalability_optimal(nof_facilities,\
                        nof_centers,\
                        nof_swaps,\
                        nof_iterations,\
                        objective,\
                        logfile,\
                        result_file)
    pprint(stats)
#end scaling_nof_facilities()


def test_batch_scaling_optimal(results_dir, test_run):
    for strategy in ['local_search_2', 'kmedoid']:
        if test_run:
            range_data = [10, 30, 100]
        else:
            range_data = [10, 100, 1000, 10000, 100000]
        scaling_nof_facilities_optimal(range_data, 'kmedian' if strategy == 'local_search_2' else 'kmeans', results_dir, strategy, test_run)

        if test_run:
            range_data = [1, 2, 3]
        else:
            range_data = [4, 5, 6, 7, 8, 9, 10, 11, 12]

        scaling_nof_centers_optimal(range_data, 'kmedian' if strategy == 'local_search_2' else 'kmeans', results_dir, strategy, test_run)

    #range_data = [1, 2, 3]
    #LS_scaling_nof_swaps(range_data, objective, results_dir)
#end test_batch_scaling_local_search()
###############################################################################


# bicriteria approximation
def test():
    N = 10
    seed = 1234

    dist_matrix = generator.get_distance_matrix(N, seed)
    print(dist_matrix)
#end test()

###############################################################################
def cmd_parser():
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group('Arguments')
    g.add_argument('-batch', '--batch-type', nargs='?', required=False, type=str,\
                   default='feasibility', choices=['feasibility', 'optimal', 'bicriteria'])
    g.add_argument('-obj', '--objective', nargs='?', required=False, type=str,\
                   default=None, choices=['kmedian', 'kmeans'])
    g.add_argument('-results', '--results-dir', nargs='?', required=False, type=str,\
                   default='exp-results')
    g.add_argument('-test', '--test-run', action='store_true')
    return parser
#end cmd_parser()



def main():
    parser      = cmd_parser()
    opts        = vars(parser.parse_args())
    batch_type  = opts['batch_type']
    objective   = opts['objective']
    results_dir = opts['results_dir']
    test_run    = opts['test_run']

    if batch_type == 'bicriteria' and objective == None:
        sys.stderr.write("Error: specify objective function\n")
        sys.exit()
    #end if

    # create directory to store experimental results
    now = datetime.now()
    results_dir_path = '%s/%s-%s-%s_%s-%s-%s'%\
                  (results_dir, now.year, now.month, now.day, now.hour, now.minute, now.second)
    cmd = 'mkdir -p %s'%(results_dir_path) 
    call(cmd, shell = True)

    # schedule experiments
    if batch_type == 'feasibility':
        test_batch_scaling(objective, results_dir_path, test_run)
    elif batch_type == 'optimal':
        test_batch_scaling_optimal(results_dir_path, test_run)
    elif batch_type == 'bicriteria':
        test_batch_scaling(objective, results_dir_path, test_run)
        test_batch_scaling_optimal(results_dir_path)
    # test_batch_scaling_LP_search(objective, results_dir_path, test_run)
    #test()
#end main()

## program entry point
if __name__=="__main__":
    main()
# main()
