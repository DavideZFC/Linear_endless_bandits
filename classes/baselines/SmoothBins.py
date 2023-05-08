from classes.linucb import linUBC
from classes.misspec import misSpec
import numpy as np
from classes.bases import *

class Bin:
    def __init__(self, start, end, d, arms):
        self.arms = arms
        self.my_arms = self.arms[start : end]
        self.n_my_arms = len(self.my_arms)
        self.d = d

        print('Bin created with arms')
        print(self.arms[start : end])
        self.linarms = make_poly_arms(self.n_my_arms, self.d, self.my_arms)



class SmoothBins:
    def __init__(self, arms, d, bins, lam=1, T=10000, m=1, epsilon=0):
        # dimension of the problem
        self.d = d

        # arms avaiable
        self.arms = arms
        self.n_arms = len(self.arms)

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
                self.learners.append(Bin(idx_start, i, self.d, self.arms))
                bin_end += 1
                idx_start = i
        # create last bin
        self.learners.append(Bin(idx_start, self.n_arms, self.d, self.arms))




