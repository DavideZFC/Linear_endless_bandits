import numpy as np
from classes.lipschitz_env import Lipschitz_Environment
from classes.baselines.advanced_learners import IGP_UCB, BPE
import matplotlib.pyplot as plt
from functions.test_algorithm import test_algorithm
from functions.confidence_bounds import bootstrap_ci
from functions.plot_from_dataset import plot_data
import os

import datetime
from joblib import Parallel, delayed
import time
import json

save = True

curve = 'gaussian'
tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
dir = 'results/'+'_'+tail+curve+'IGPvsBPE'
# dir = 'data/davide/RankingBandit'+'_'+tail

os.mkdir(dir)
dir = dir+'/'


print("-----------------------------------------------------------\n")
print("Linear Endless Bandit Performances on Synthetic Data\n")
print("-----------------------------------------------------------\n")


env = Lipschitz_Environment(lim=1.0, sigma=1.0, curve = curve, n_arms=100)
T = 1000
seeds = 10


policies = [IGP_UCB(arms=env.x, T=T), BPE(env.x, T=T)]
labels = ['IGP-UCB', 'BPE']

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
    plot_data(np.arange(0,T), low, high, col='C{}'.format(i), label=labels[i])

    # save data in given folder
    np.save(dir+labels[i], results[i])


with open(dir+"running_times.json", "w") as f:
    # Convert the dictionary to a JSON string and write it to the file
    json.dump(running_times, f)

plt.legend()
plt.title('Regret curves')
plt.savefig(dir+'regret_plot.pdf')

