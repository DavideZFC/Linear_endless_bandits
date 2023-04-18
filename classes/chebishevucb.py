from classes.linucb import linUBC
import numpy as np
from math import comb

def fatt(n):
    if not type(n)==int:
        raise TypeError('n must be integer')
    if n < 0:
        raise ValueError('n must be nonnegative, here is '+str(n))
    if n == 0:
        return 1
    else:
        return fatt(n-1)


class ChebishevUCB:
    def __init__(self, arms, d, lam=1, T=10000, m=1, only_even=False):
        # dimension of the problem
        self.d = d

        # arms avaiable
        self.arms = arms
        self.n_arms = len(self.arms)

        # time horizon
        self.T = T

        # parameter so see if we have an even function
        self.even = only_even

        # initialize learner
        self.make_linUCB(lam, m)

    def reset(self):
        self.learner.reset()

    
    def make_chebishev_arms(self):
        self.linUCBarms = np.zeros((self.n_arms, self.d))

        # build linear features from the arms
        for j in range(self.d):
            self.linUCBarms[:,j] = np.cos(j*np.arccos(self.arms))


    def make_chebishev_even_arms(self):
        self.linUCBarms = np.zeros((self.n_arms, self.d))

        # build linear features from the arms
        for j in range(self.d):
            self.linUCBarms[:,j] = np.cos((2*j)*np.arccos(self.arms))

        
    def make_linUCB(self, lam, m):

        # prepare feature matrix
        if self.even:
            self.make_chebishev_even_arms()
        else:
            self.make_chebishev_arms()

        # initialize linUCB
        print('Instance linUCB with T='+str(self.T))
        self.learner = linUBC(self.linUCBarms, lam=lam, T=self.T, m=m)

    def pull_arm(self):
        
        # ask what arm to pull
        _, arm = self.learner.pull_arm()

        return arm
    
    def update(self, arm, reward):

        # pass from arm index to arm vector
        arm_vector = self.linUCBarms[arm,:]

        # update base learner
        self.learner.update(arm_vector, reward)
