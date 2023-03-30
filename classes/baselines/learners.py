import numpy as np

class UCB1():
    '''
    Classical UCB1 algorithm
    '''
    def __init__(self, N=200):
        self.N = N
        self.rewards = np.zeros(N)
        self.times_pulled = np.zeros(N)
        self.t = 0

    def pull_arm(self):
        if self.t < self.N:
            arm = self.t
        else:
            self.empirical_means = self.rewards / self.times_pulled
            confidence = np.sqrt(2*np.log(self.t)/self.times_pulled)
            upper_bound = self.empirical_means + confidence
            arm = np.argmax(upper_bound)
        return arm

    def update(self, arm, reward):
        self.t += 1
        self.times_pulled[arm] += 1
        self.rewards[arm] += reward

    def reset(self):
        self.rewards = np.zeros(self.N)
        self.times_pulled = np.zeros(self.N)
        self.t = 0


class TS():
    '''
    Classical Thompson Sampling algorithm
    '''
    def __init__(self, N=200):
        self.N = N
        self.rewards = np.zeros(N)
        self.times_pulled = np.zeros(N)
        self.t = 0

    def pull_arm(self):
        if self.t < self.N:
            arm = self.t
        else:
            samples = np.random.beta(self.rewards+1, self.times_pulled-self.rewards+1)
            arm = np.argmax(samples)
        return arm

    def update(self, arm, reward):
        self.t += 1
        self.times_pulled[arm] += 1
        self.rewards[arm] += reward

    def reset(self):
        self.rewards = np.zeros(self.N)
        self.times_pulled = np.zeros(self.N)
        self.t = 0







