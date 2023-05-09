from classes.linucb import linUBC
from classes.misspec import misSpec
import numpy as np
from classes.bases import *

class Bin:
    def __init__(self, start, end, d, arms, T=1000, lam=1., m=1, epsilon=0.01):
        self.arms = arms
        self.my_arms = self.arms[start : end]
        self.n_my_arms = len(self.my_arms)
        self.d = d

        # Hyper parameters
        self.lam = lam
        self.m = m
        self.epsilon = epsilon
        self.T = T

        # it is important to store the first index so that when we decide which arm to pull, 
        # we can pass the right arm to the external learner
        self.first_index = start
        self.last_index = end

        # print('Bin created with arms')
        # print(self.arms[start : end])
        self.linarms = make_poly_arms(self.n_my_arms, self.d, self.my_arms)

        self.make_misSpec()


    def make_misSpec(self):
        # initialize linUCB
        self.learner = misSpec(self.linarms, lam=self.lam, T=self.T, m=self.m, epsilon=self.epsilon, C1=10)

    def pull_arm(self):        
        # ask what arm to pull
        _, arm = self.learner.pull_arm()
        return self.first_index + arm
    
    def get_upper_bound(self):
        return self.learner.upper_bound
    
    def update(self, arm, reward):
        if (self.first_index <= arm ) and (arm <= self.last_index):
            arm = arm - self.first_index
        else:
            return
        # pass from arm index to arm vector
        arm_vector = self.linarms[arm,:]
        # update base learner
        self.learner.update(arm_vector, reward)



class SmoothBins:
    def __init__(self, arms, d, bins, lam=1, T=10000, m=1, epsilon=0):
        # dimension of the problem
        self.d = d

        # arms avaiable
        self.arms = arms
        self.n_arms = len(self.arms)

        # hyperparameters
        self.lam = lam
        self.m = m

        # time horizon
        self.T = T
        self.t = 0

        # bins
        self.bins = bins
        self.bin_delimiter = np.linspace(-1,1,self.bins+1)

        # instantiate learner for each bin
        self.learners = []

        # make the arms
        idx_start = 0
        bin_end = 1
        for i in range(self.n_arms):
            if self.arms[i] > self.bin_delimiter[bin_end]:
                self.learners.append(Bin(idx_start, i, self.d, self.arms, T=self.T, m=self.m, lam=self.lam))
                bin_end += 1
                idx_start = i
        # create last bin
        self.learners.append(Bin(idx_start, self.n_arms, self.d, self.arms, T=self.T, m=self.m, lam=self.lam))

    def reset(self):
        ''' re-create the bins, one by one'''

        self.learners = []

        # make the arms
        idx_start = 0
        bin_end = 1
        for i in range(self.n_arms):
            if self.arms[i] > self.bin_delimiter[bin_end]:
                self.learners.append(Bin(idx_start, i, self.d, self.arms, T=self.T, m=self.m, lam=self.lam))
                bin_end += 1
                idx_start = i
        # create last bin
        self.learners.append(Bin(idx_start, self.n_arms, self.d, self.arms, T=self.T, m=self.m, lam=self.lam))

    def pull_arm(self):

        if self.t < self.bins:
            arm = self.learners[self.t].pull_arm()
            return arm
    
        else:
            # choose the bin with maximum upper bound
            upper_bounds = np.zeros(self.bins)
            for j in range(self.bins):
                upper_bounds[j] = self.learners[j].get_upper_bound()

            best_bin = np.argmax(upper_bounds)
            arm = self.learners[best_bin].pull_arm()
            return arm
        
    def update(self, arm, reward):
        for l in self.learners:
            l.update(arm, reward)







