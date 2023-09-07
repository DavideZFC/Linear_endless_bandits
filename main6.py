import fire
import numpy as np
from classes.lipschitz_env import Lipschitz_Environment
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

class Experiment(object):
   def make(self, curve):

        print(curve)

        # curve = 'critical_poly'
        tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
        dir = 'results/'+'_'+tail+curve

        os.mkdir(dir)
        dir = dir+'/'


        env = Lipschitz_Environment(lim=1.0, sigma=1.0, curve = curve, n_arms=100)
        T = 10000
        seeds = 20
        # save reward curve
        np.save(dir+'reward_curve', env.y)

        dp = 9

        policies = [MetaLearner('Poly', env.x, dp, T=T, m=1.0), MetaLearner('Legendre', env.x, dp, T=T, m=1.0), MetaLearner('Chebishev', env.x, dp, T=T, m=1.0)]
        labels = ['Poly', 'Legendre', 'Chebishev']


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





if __name__ == '__main__':
  fire.Fire(Experiment)