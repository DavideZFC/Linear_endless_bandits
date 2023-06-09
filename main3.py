import numpy as np
from classes.lipschitz_env import Lipschitz_Environment
from classes.fourierucb import FourierUCB
from classes.legendreucb import LegendreUCB
from classes.chebishevucb import ChebishevUCB
from classes.baselines.learners import UCB1
from classes.baselines.lips_learners import ZOOM
from classes.baselines.advanced_learners import IGP_UCB
from classes.meta_learner import MetaLearner
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

curve = 'critical_poly'
tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
dir = 'results/'+'_'+tail+curve
# dir = 'data/davide/RankingBandit'+'_'+tail

os.mkdir(dir)
dir = dir+'/'


print("-----------------------------------------------------------\n")
print("Linear Endless Bandit Performances on Synthetic Data\n")
print("-----------------------------------------------------------\n")


env = Lipschitz_Environment(lim=1.0, sigma=1.0, curve = curve, n_arms=100)
T = 10000
seeds = 20

dp = 9

policies = [MetaLearner('Poly', env.x, dp, T=T, m=1.0), MetaLearner('Legendre', env.x, dp, T=T, m=1.0), MetaLearner('Chebishev', env.x, dp, T=T, m=1.0)]
labels = ['Poly', 'Legendre', 'Chebishev']

'''
############# Fourier
mf = 0.1
df = 8
policies += [FourierUCB(env.x, df, T, m=mf), FourierUCB(env.x, df, T, m=mf, only_even=True)]
labels += ['FourierUCB', 'FourierUCB_even']

############# Legendre
ml = 1.0
dl = 8
policies += [LegendreUCB(env.x, dl, T=T, m=ml), LegendreUCB(env.x, dl, T=T, m=ml, only_even=True)]
labels += ['LegendreUCB', 'LegendreUCB_even']

parameters = {'mf': mf, 'df': df, 'ml': ml, 'dl': dl}
'''
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

with open(dir+"parameters.json", "w") as g:
    # Convert the dictionary to a JSON string and write it to the file
    try:
        json.dump(parameters, g)
    except:
        pass

plt.legend()
plt.title('Regret curves')
plt.savefig(dir+'regret_plot.pdf')

