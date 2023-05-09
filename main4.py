import numpy as np
from classes.lipschitz_env import Lipschitz_Environment
from classes.baselines.SmoothBins import SmoothBins
from classes.baselines.learners import UCB1
import matplotlib.pyplot as plt
from functions.test_algorithm import test_algorithm
from functions.confidence_bounds import bootstrap_ci
from functions.plot_from_dataset import plot_data
import os

import datetime
import time
import json

save = False

curve = 'gaussian'
tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
dir = 'results/'+'_'+tail+curve

if save:
    os.mkdir(dir)
    dir = dir+'/'

T = 5000
seeds = 5
 
env = Lipschitz_Environment(lim=1.0, sigma=1.0, curve = curve, n_arms=int(T**(1/2)))
env.plot_curve()


policies = [SmoothBins(env.x, d=3, bins=5, T=T), UCB1(len(env.x))]#,GPUCB(arms=env.x, update_every=5, kernel='dirichlet'), GPUCB(arms=env.x, update_every=5), FourierUCB(env.x, df, T, m=mf), LegendreUCB(env.x, dl, T=T, m=ml), ChebishevUCB(env.x, dc, T=T, m=mc), FourierUCB(env.x, df, T, m=mf, only_even=True), LegendreUCB(env.x, dl, T=T, m=ml, only_even=True), ChebishevUCB(env.x, dc, T=T, m=mc, only_even=True)] 
labels = ['SmoothBins', 'UCB1']#,  'LegendreUCB', 'EvenFourier', 'EvenLegendre', 'EvenChebishev']#, 'ZOOM', 'GPTS', 'GaussUCB', 



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

if save:
    with open(dir+"running_times.json", "w") as f:
        # Convert the dictionary to a JSON string and write it to the file
        json.dump(running_times, f)

plt.legend()

if save:
    plt.title('Regret curves')
    plt.savefig(dir+'regret_plot.pdf')
plt.show()