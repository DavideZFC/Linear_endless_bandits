import numpy as np

from classes.lipschitz_env import *
from classes.lipschitz_env import Lipschitz_Environment
from classes.lipschitz_env import Lipschitz_Environment
from classes.fourierucb import FourierUCB
from classes.legendreucb import LegendreUCB
from classes.chebishevucb import ChebishevUCB
from functions.test_algorithm import test_algorithm
from functions.confidence_bounds import bootstrap_ci
from functions.plot_from_dataset import plot_data
import os

import datetime
from joblib import Parallel, delayed
import time
import json



curves = ['gaussian', 'cosine', 'even_poly', 'poly', 'sin-like', 'spike', 'random']

for curve in curves:
    tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
    dir = dir = 'results/'+'_'+tail+'Y'+curve

    os.mkdir(dir)
    dir = dir+'/'


    print("-----------------------------------------------------------\n")
    print("Linear Endless Bandit Performances on Synthetic Data\n")
    print("-----------------------------------------------------------\n")


    env = Lipschitz_Environment(lim=1.0, sigma=1.0, curve = curve, n_arms=100)

    # save reward curve
    np.save(dir+'reward_curve', env.y)

    T = 10000
    seeds = 20

    policies = []
    labels = []

    ############# Fourier
    mf = 0.1
    df = 4
    policies += [FourierUCB(env.x, df, T, m=mf), FourierUCB(env.x, df, T, m=mf, only_even=True)]
    labels += ['FourierUCB', 'FourierUCB_even']

    ############# Legendre
    ml = 0.1
    dl = 5
    policies += [LegendreUCB(env.x, dl, T=T, m=ml), LegendreUCB(env.x, dl, T=T, m=ml, only_even=True)]
    labels += ['LegendreUCB', 'LegendreUCB_even']

    parameters = {'mf': mf, 'df': df, 'ml': ml, 'dl': dl}

    ############# ChebishevUCB
    mc = 0.05
    dc = 5
    policies += [ChebishevUCB(env.x, dl, T=T, m=ml), ChebishevUCB(env.x, dl, T=T, m=ml, only_even=True)]
    labels += ['ChebishevUCB', 'ChebishevUCB_even']

    parameters = {'mf': mf, 'df': df, 'ml': ml, 'dl': dl, 'mc' : mc, 'dc' : dc}

    running_times = {}

    results = [] 

    for i in range(len(policies)):

        ####################################
        # actual algorithm simulation

        # evaluate running time of the algorithm
        t0 = time.time()

        # test the algorithm
        results.append(Parallel(n_jobs=seeds)(delayed(test_algorithm)(policies[i], env, T, seeds=1, first_seed=seed) for seed in range(seeds)))

        # store time
        t1 = time.time()
        running_times[labels[i]] = t1 - t0
        
        print(labels[i] + ' finished')

        ####################################
        # part to save data

        results[i] = np.concatenate(results[i], axis=0)

        # make nonparametric confidence intervals
        low, high = bootstrap_ci(results[i])

        # make plot
        # plot_data(np.arange(0,T), low, high, col='C{}'.format(i), label=labels[i])

        # save data in given folder
        np.save(dir+labels[i], results[i])


    with open(dir+"running_times.json", "w") as f:
        # Convert the dictionary to a JSON string and write it to the file
        json.dump(running_times, f)

    with open(dir+"parameters.json", "w") as g:
        # Convert the dictionary to a JSON string and write it to the file
        json.dump(parameters, g)