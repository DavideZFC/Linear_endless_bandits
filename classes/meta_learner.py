from classes.linucb import linUBC
from classes.misspec import misSpec
import numpy as np
from classes.bases import *


class MetaLearner:
    def __init__(self, basis, arms, d, lam=1, T=10000, m=1, only_even=False, miss=False, epsilon=0):
        # dimension of the problem
        self.d = d

        # arms avaiable
        self.arms = arms
        self.n_arms = len(self.arms)

        # time horizon
        self.T = T

        # parameter so see if we have an even function
        self.even = only_even
  
        # choose the basis
        if basis == 'Fourier':
            if only_even:
                self.linarms = make_cosin_arms(self.n_arms, d, arms)
            else:
                self.linarms = make_sincos_arms(self.n_arms, d, arms)
            print('Fourier basis init')

        elif basis == 'Legendre':
            if only_even:
                self.linarms = make_legendre_even_arms(self.n_arms, d, arms)
            else:
                self.linarms = make_legendre_arms(self.n_arms, d, arms)
            print('Legendre basis init')

        elif basis == 'Chebishev':
            if only_even:
                self.linarms = make_chebishev_even_arms(self.n_arms, d, arms)
            else:
                self.linarms = make_chebishev_arms(self.n_arms, d, arms)
            print('Chebishev basis init')  

        else:
            raise Exception("Sorry, basis not found")

        # initialize learner
        if miss:
            self.make_misSpec(lam, m, epsilon)
        else:
            self.make_linUCB(lam, m)       


    def reset(self):
        self.learner.reset()

    def make_linUCB(self, lam, m):
        # initialize linUCB
        self.learner = linUBC(self.linarms, lam=lam, T=self.T, m=m)

    def make_misSpec(self, lam, m, epsilon):
        # initialize linUCB
        self.learner = misSpec(self.linarms, lam=lam, T=self.T, m=m, epsilon=epsilon, C1=10)


    def pull_arm(self):        
        # ask what arm to pull
        _, arm = self.learner.pull_arm()
        return arm
    
    def update(self, arm, reward):
        # pass from arm index to arm vector
        arm_vector = self.linarms[arm,:]
        # update base learner
        self.learner.update(arm_vector, reward)

    


