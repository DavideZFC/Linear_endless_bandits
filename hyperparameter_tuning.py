import numpy as np
from classes.lipschitz_env import Lipschitz_Environment
from classes.meta_learner import MetaLearner
from classes.baselines.SmoothBins import SmoothBins
from functions.test_algorithm import test_algorithm
from functions.confidence_bounds import bootstrap_ci
from functions.plot_from_dataset import plot_data
import os

import datetime
from joblib import Parallel, delayed
import time
import json

T = 20
seeds = 5

curve = 'gaussian' 
env = Lipschitz_Environment(lim=1.0, sigma=1.0, curve = curve, n_arms=int(T**(1/2)))

tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
dir = 'results/'+'_'+tail+curve+'HT'
os.mkdir(dir)
dir = dir+'/'

##########################
### tune parameters of MetaLearner
##########################

bases = ['Fourier', 'Legendre', 'Chebishev']
d_list = [4, 5, 6, 8, 10, 12]
m_list = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]



for basis in bases:
    print('start with basis '+basis)

    param_dic = {}

    for d in d_list:
        for m in m_list:
            key = 'd={} m={}'.format(d,m)

            policy = MetaLearner(basis=basis, arms=env.x, d=d, T=T, m=m)
            regret = test_algorithm(policy, env, T, seeds)

            results = Parallel(n_jobs=seeds)(delayed(test_algorithm)(policy, env, T, seeds=1, first_seed=seed) for seed in range(seeds))
            regret = np.concatenate(results, axis=0)

            mean_regret = np.mean(regret[:,-1])

            param_dic[key] = mean_regret


    with open(dir+basis+".json", "w") as f:
        # Convert the dictionary to a JSON string and write it to the file
        json.dump(param_dic, f)
    f.close()

##############################
# tune parameters of SmoothBins
##############################

d_list = [4, 5, 6, 8, 10, 12]
m_list = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
bins_list = [5, 8, 10, 20]
epsilon_list = [0.0, 0.01, 0.05, 0.1]

# number of different HP to try
N_trials = 50

par_dic = {}

random_d = np.random.randint(len(d_list), size=N_trials)
random_m = np.random.randint(len(m_list), size=N_trials)
random_bins = np.random.randint(len(bins_list), size=N_trials)
random_epsilon = np.random.randint(len(epsilon_list), size=N_trials)

print('start with SmoothBin')
for t in range(N_trials):
    d = d_list[random_d[t]]
    m = m_list[random_m[t]]
    bins = bins_list[random_bins[t]]
    epsilon = epsilon_list[random_epsilon[t]]

    key = 'd={} m={} b={} e={}'.format(d,m,bins,epsilon)

    policy = SmoothBins(env.x, d=d, bins=bins, T=T, m=m, epsilon=epsilon)

    regret = test_algorithm(policy, env, T, seeds)
    mean_regret = np.mean(regret[:,-1])

    par_dic[key] = mean_regret

with open(dir+"smooth_bins.json", "w") as f:
    # Convert the dictionary to a JSON string and write it to the file
    json.dump(par_dic, f)

f.close()