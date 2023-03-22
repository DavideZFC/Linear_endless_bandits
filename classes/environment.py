import numpy as np

class Environment:
    def __init__(self, theta, sd=0.1):
        self.theta = theta
        self.sd = sd
    
    def pull_arm(self, arm):
        return np.dot(arm, self.theta) + np.random.normal(0,self.sd)