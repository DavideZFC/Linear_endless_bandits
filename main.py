import numpy as np
from classes.environment import Environment
from classes.lipschitz_env import Lipschitz_Environment
from classes.linucb import linUBC
from classes.lipucb import lipUCB
import matplotlib.pyplot as plt


d = 5
n_arms = 100
T = 1000
theta = np.array([0., 1., 0., 0., 0.])

env = Environment(theta)

arms = np.random.normal(size=(n_arms,d))
policy = linUBC(arms_matrix=arms, T=T)
opt = env.get_optimum(arms)

reward_vector = np.zeros(T)

for t in range(T):
    arm, _ = policy.pull_arm()
    reward = env.pull_arm(arm)
    reward_vector[t] = reward
    policy.update(arm, reward)

plt.plot(reward_vector)
plt.axhline(y=opt, color = 'C1', linestyle='dashed')
plt.show()