from classes.linucb import linUBC
import numpy as np

class lipUCB:
    def __init__(self, arms, d, lam=1, T=10000, m=1):
        # dimension of the problem
        self.d = d

        # arms avaiable
        self.arms = arms
        self.n_arms = len(self.arms)

        # time horizon
        self.T = T

        # initialize learner
        self.make_linUCB(lam, m)

    def reset(self):
        self.learner.reset()


    def make_cosin_arms(self):
        self.linUCBarms = np.zeros((self.n_arms, self.d))

        # build linear features from the arms
        for j in range(self.d):
            self.linUCBarms[:,j] = np.cos(np.pi*j*self.arms)

        
    def make_linUCB(self, lam, m):

        # prepare feature matrix
        self.make_cosin_arms()

        # initialize linUCB
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

