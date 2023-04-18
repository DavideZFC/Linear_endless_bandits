import numpy as np
import matplotlib.pyplot as plt
from classes.legendreucb import LegendreUCB
from classes.chebishevucb import ChebishevUCB

N = 100

x = np.linspace(-1,1,N)

policy = ChebishevUCB(arms=x, d=5)

arms = policy.linUCBarms

for i in range(arms.shape[1]):
    plt.plot(x, arms[:,i])

plt.show()


