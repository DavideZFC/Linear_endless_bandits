import numpy as np
from classes.environment import Environment
from classes.linucb import linUBC

d = 5
N = 100
T = 100
theta = np.random.normal(size=d)
print(theta)

env = Environment(theta)
arms = np.random.normal(size=(N,d))
policy = linUBC(arms_matrix=arms, T=T)

for t in range(T):
    print(np.sum((theta.reshape(-1,1)-policy.estimate_theta())**2))
    reward = env.pull_arm(arms[t,:])
    policy.update(arms[t,:], reward)