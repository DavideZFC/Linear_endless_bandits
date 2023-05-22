import numpy as np
from classes.lipschitz_env import Lipschitz_Environment
from classes.fourierucb import FourierUCB
from classes.legendreucb import LegendreUCB
from classes.chebishevucb import ChebishevUCB
from classes.baselines.learners import UCB1
from classes.baselines.lips_learners import ZOOM
from classes.baselines.IGPUCB import IGPUCB
from classes.baselines.UCBMetaAlgorithm import UCBMetaAlgorithm
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

print('Which experiment (a,b,c,d) to perform?')
e = input()
if e == 'a':
    curve = 'gaussian'
elif e == 'b':
    curve = 'even_poly'
elif e == 'c':
    curve = 'sin-like'
elif e == 'd':
    curve = 'spike'
else:
    raise Exception("Environment not supported")

print('Do you want to use even basis function? y/n')
only_even = input()
only_even = (only_even == 'y')

tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
dir = 'results/'+'_'+tail+curve

os.mkdir(dir)
dir = dir+'/'


print("-----------------------------------------------------------\n")
print("Orthogonal Function Representations for Continuous Armed Bandits on Synthetic Data\n")
print("-----------------------------------------------------------\n")


env = Lipschitz_Environment(lim=1.0, sigma=1.0, curve = curve, n_arms=100)
T = 1000
seeds = 5

policies = [ZOOM(env.x), IGPUCB(env.x, T,B=4, R=1, update_every=10), UCBMetaAlgorithm(env.x, d=5, bins=5, T=T, m=2, lam=1, epsilon=0.1)]
labels = ['ZOOM', 'GP', 'UMA']

############# Fourier
mf = 0.1
df = 8
policies += [FourierUCB(env.x, df, T, m=mf, only_even=only_even)]
if only_even:
    labels += ['FourierUCB+E']
else:
    labels += ['FourierUCB']

############# Legendre
ml = 1.0
dl = 6
policies += [LegendreUCB(env.x, dl, T=T, m=ml, only_even=only_even)]
if only_even:
    labels += ['LegendreUCB+E']
else:
    labels += ['LegendreUCB']

############# Chebyshev
ml = 1.0
dl = 6
policies += [ChebishevUCB(env.x, dl, T=T, m=ml, only_even=only_even)]
if only_even:
    labels += ['ChebyshevUCB+E']
else:
    labels += ['ChebyshevUCB']

parameters = {'mf': mf, 'df': df, 'ml': ml, 'dl': dl}

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
    json.dump(parameters, g)

plt.legend()
plt.title('Regret curves')
plt.savefig(dir+'regret_plot.pdf')
plt.show()

