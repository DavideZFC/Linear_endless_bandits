import numpy as np
from classes.lipschitz_env import Lipschitz_Environment
from classes.fourierucb import FourierUCB
from classes.legendreucb import LegendreUCB
from classes.baselines.learners import UCB1
from classes.baselines.lips_learners import ZOOM
from classes.baselines.advanced_learners import Gauss_Bandit, GPTS, IGP_UCB
import matplotlib.pyplot as plt
from functions.test_algorithm import test_algorithm
from functions.confidence_bounds import bootstrap_ci
from functions.plot_from_dataset import plot_data
import os

import datetime
import time
import json

save = True

curve = 'gaussian'
tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
dir = 'results/'+'_'+tail+curve

if save:
    os.mkdir(dir)
    dir = dir+'/'
 
env = Lipschitz_Environment(lim=1.0, sigma=0.5, curve = curve, n_arms=100)
env.plot_curve()

T = 10000
seeds = 5

# Fourier parameters
mf = 0.1
df = 8

# Legendre parameters
ml = 1.0
dl = 6

m_list = [1.0, 10.0]
d_list = [8, 12]
lambda_list = [10.0, 100.0]

policies = [UCB1(len(env.x)), FourierUCB(env.x, df, T, m=mf), LegendreUCB(env.x, dl, T=T, m=ml), FourierUCB(env.x, df, T, m=mf, only_even=True), LegendreUCB(env.x, dl, T=T, m=ml, only_even=True)]# IGP_UCB(env.x, T), IGP_UCB(env.x, T, update_every=10), ZOOM(env.x), GPTS(env.x), Gauss_Bandit(env.x), 
labels = ['UCB1', 'FourierUCB', 'LegendreUCB', 'EvenFourier', 'EvenLegendre']#, 'ZOOM', 'GPTS', 'GaussUCB', 

'''
for mu in m_list:
    for di in d_list:
        for lam in lambda_list:
            policies.append(LegendreUCB(env.x, di, lam=lam, T=T, m=mu))
            labels.append('Legendre_m={}_d={}_lam={}'.format(mu,di,lam))
'''

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
