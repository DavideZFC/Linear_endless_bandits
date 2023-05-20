import numpy as np
import matplotlib.pyplot as plt
from classes.legendreucb import LegendreUCB
from classes.chebishevucb import ChebishevUCB
from classes.fourierucb import FourierUCB
from classes.meta_learner import MetaLearner

N = 100

x = np.linspace(-1,1,N)

# policies = [FourierUCB(arms=x, d=5), LegendreUCB(arms=x, d=5), ChebishevUCB(arms=x, d=5)]
policies = [MetaLearner(arms=x, d=5, basis='Poly'), MetaLearner(arms=x, d=5, basis='Legendre'), MetaLearner(arms=x, d=5, basis='Chebishev')]

# fig, ax = plt.subplots(1,3,figsize=(24,6))
fig, ax = plt.subplots(1,3,figsize=(12,3))

for j in range(3):
    policy = policies[j]
    arms = policy.linarms #policy.linUCBarms

    for i in range(arms.shape[1]):
        ax[j].plot(x, arms[:,i], label='n={}'.format(i))
        ax[j].grid(True)
        ax[j].legend(loc='upper left')
    
   
fig.show()
fig.savefig('bases.pdf')


