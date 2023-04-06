import numpy as np
from classes.environment import Environment
from classes.lipschitz_env import Lipschitz_Environment
from classes.linucb import linUBC
from classes.lipucb import lipUCB
import matplotlib.pyplot as plt

import time
start = time.time()

d = 5
n_arms = 100
T = 20000
m = 100
theta = np.array([0.9, -1., 1., -1., 1.1])

env = Environment(theta, sd=0.2)

arms = np.random.normal(size=(n_arms,d))
policy = linUBC(arms_matrix=arms, T=T, m=m)
opt = env.get_optimum(arms)

reward_vector = np.zeros(T)

for t in range(T):
    arm, _ = policy.pull_arm()
    reward = env.pull_arm(arm)
    reward_vector[t] = reward
    policy.update(arm, reward)

end = time.time()

print('Tempo trascorso = {}'.format(end-start))

plt.plot(reward_vector)
plt.axhline(y=opt, color = 'C1', linestyle='dashed')
plt.show()