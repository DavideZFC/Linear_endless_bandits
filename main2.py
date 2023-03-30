import numpy as np
from classes.lipschitz_env import Lipschitz_Environment
from classes.lipucb import lipUCB
from classes.baselines.learners import UCB1
from classes.baselines.lips_learners import ZOOM
from classes.baselines.advanced_learners import Gauss_Bandit, GPTS
import matplotlib.pyplot as plt
from functions.test_algorithm import test_algorithm
from functions.confidence_bounds import bootstrap_ci
from functions.plot_from_dataset import plot_data
 
env = Lipschitz_Environment(lim=1.0, curve = 'gaussian', n_arms=100)
env.plot_curve()

T = 4000
d = 8
seeds = 20

# m_vals = [0.01, 0.02, 0.05, 0.1]
policies = [GPTS(env.x), Gauss_Bandit(env.x), UCB1(len(env.x)), ZOOM(env.x), lipUCB(env.x, d, T, m=0.1)]
labels = ['GPTS', 'GaussUCB', 'UCB1', 'ZOOM', 'LipUCB']

for i in range(len(policies)):
    regret = test_algorithm(policies[i], env, T, seeds)
    print(labels[i] + ' finished')
    low, high = bootstrap_ci(regret)
    plot_data(np.arange(0,T), low, high, col='C{}'.format(i), label=labels[i])

plt.legend()
plt.show()
