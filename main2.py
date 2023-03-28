import numpy as np
from classes.lipschitz_env import Lipschitz_Environment
from classes.lipucb import lipUCB
import matplotlib.pyplot as plt

env = Lipschitz_Environment(lim=1.0, curve = 'cosine', n_arms=100)
env.plot_curve()

T = 1000
d = 5
m = 2
policy = lipUCB(env.x, d, T, m=m)
reward_vector = np.zeros(T)

opt = env.get_optimum()

for t in range(T):
    arm = policy.pull_arm()
    reward = env.pull_arm(arm)
    reward_vector[t] = reward
    policy.update(arm, reward)

plt.plot(reward_vector)
plt.axhline(y=opt, color = 'C1', linestyle='dashed')
plt.show()
