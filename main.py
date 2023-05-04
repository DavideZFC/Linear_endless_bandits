import numpy as np
from classes.environment import Environment
from classes.misspec import misSpec
from classes.linucb import linUBC
import matplotlib.pyplot as plt

import time


d = 5
n_arms = 100
T = 10000
m = 100
theta = np.array([0.9, -1., 1., -1., 1.1])

env = Environment(theta, sd=0.1)

arms = np.random.normal(size=(n_arms,d))
policy = linUBC(arms_matrix=arms, T=T, m=m)
opt = env.get_optimum(arms)

reward_vector = np.zeros(T)

# measure time of LinUCB
start = time.time()

for t in range(T):
    arm, _ = policy.pull_arm()
    reward = env.pull_arm(arm)
    reward_vector[t] = reward
    policy.update(arm, reward)

end = time.time()

print('Tempo trascorso linUCB = {}'.format(end-start))

policy = misSpec(arms_matrix=arms, epsilon=0.0, sigma=0.1, C1=10)# il problema è ovviamente il C
reward_vector2 = np.zeros(T)

# measure time of Misspec
start = time.time()

for t in range(T):
    arm, _ = policy.pull_arm()
    reward = env.pull_arm(arm)
    reward_vector2[t] = reward
    policy.update(arm, reward)

end = time.time()
print('Tempo trascorso misSPEC = {}'.format(end-start))



plt.plot(reward_vector, label='LinUCB')
plt.plot(reward_vector2, label='misSPEC')
plt.axhline(y=opt, color = 'C1', linestyle='dashed')
plt.legend()
plt.show()