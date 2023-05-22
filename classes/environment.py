import numpy as np

class Environment:
    '''
    Environment to use linear bandit algorithms, no related so smooth bandits
    '''
    def __init__(self, theta, sd=0.1, epsilon=0):
        self.theta = theta
        self.sd = sd
        self.epsilon = 0
    
    def pull_arm(self, arm):
        return np.dot(arm, self.theta) + np.random.normal(0,self.sd) + self.epsilon
    
    def get_optimum(self, arms):
        means = np.matmul(arms, self.theta.reshape(-1, 1))
        return max(means)