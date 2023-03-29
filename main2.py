import numpy as np
from classes.lipschitz_env import Lipschitz_Environment
from classes.lipucb import lipUCB
import matplotlib.pyplot as plt
from functions.test_algorithm import test_algorithm
from functions.confidence_bounds import bootstrap_ci
from functions.plot_from_dataset import plot_data
 
env = Lipschitz_Environment(lim=1.0, curve = 'gaussian', n_arms=100)
env.plot_curve()

T = 10000
d = 8
seeds = 20

m_vals = [0.01, 0.02, 0.05, 0.1]
policies = [lipUCB(env.x, d, T, m=m) for m in m_vals]

for i in range(len(m_vals)):
    regret = test_algorithm(policies[i], env, T, seeds)
    low, high = bootstrap_ci(regret)
    plot_data(np.arange(0,T), low, high, col='C{}'.format(i), label='lipUCB_{}'.format(m_vals[i]))
plt.legend()
plt.show()
