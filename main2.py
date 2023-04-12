import numpy as np
from classes.lipschitz_env import Lipschitz_Environment
from classes.fourierucb import FourierUCB
from classes.legendreucb import LegendreUCB
from classes.baselines.learners import UCB1
from classes.baselines.lips_learners import ZOOM
from classes.baselines.advanced_learners import Gauss_Bandit, GPTS
import matplotlib.pyplot as plt
from functions.test_algorithm import test_algorithm
from functions.confidence_bounds import bootstrap_ci
from functions.plot_from_dataset import plot_data
import os

import datetime
import time
import json

save = True

curve = 'cosine'
tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
dir = 'results/'+'_'+tail+curve

if save:
    os.mkdir(dir)
    dir = dir+'/'
 
env = Lipschitz_Environment(lim=1.0, sigma=0.5, curve = curve, n_arms=100)
env.plot_curve()

T = 10000
d = 8
seeds = 5
m = 0.1
m_list = [0.01, 0.1, 1.0]
d_list = [4, 6, 8]

policies = [UCB1(len(env.x))]# LegendreUCB(env.x, d, T, m=m) , FourierUCB(env.x, d, T, m=m), ZOOM(env.x), GPTS(env.x), Gauss_Bandit(env.x), 
labels = ['UCB1']#'LegrendreUCB', 'FourierUCB', 'ZOOM', 'GPTS', 'GaussUCB', 

for mu in m_list:
    for di in d_list:
        policies.append(LegendreUCB(env.x, di, T=T, m=mu))
        labels.append('Legendre_m={}_d={}'.format(mu,di))

running_times = {}
for i in range(len(policies)):

    # evaluate running time of the algorithm
    t0 = time.time()

    # test the algorithm
    regret = test_algorithm(policies[i], env, T, seeds)

    # save regret matrix
    np.save(dir+labels[i], regret)

    # store time
    t1 = time.time()
    running_times[labels[i]] = t1 - t0
    
    print(labels[i] + ' finished')

    # prepare regret curve
    low, high = bootstrap_ci(regret)
    plot_data(np.arange(0,T), low, high, col='C{}'.format(i), label=labels[i])

with open(dir+"running_times.json", "w") as f:
    # Convert the dictionary to a JSON string and write it to the file
    json.dump(running_times, f)

plt.legend()

if save:
    plt.title('Regret curves')
    plt.savefig(dir+'regret_plot.pdf')
plt.show()
