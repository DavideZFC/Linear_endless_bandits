import numpy as np
import matplotlib.pyplot as plt
from classes.legendreucb import LegendreUCB

N = 100

x = np.linspace(-1,1,N)

policy = LegendreUCB(arms=x, d=6)

arms = policy.linUCBarms

for i in range(arms.shape[1]):
    plt.plot(arms[:,i])

plt.show()


