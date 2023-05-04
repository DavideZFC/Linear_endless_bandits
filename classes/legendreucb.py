from classes.linucb import linUBC
import numpy as np
from math import comb

# TODO normalizzare i polinomi in L2

class LegendreUCB:
    def __init__(self, arms, d, lam=1, T=10000, m=1, only_even=False, exp=1):
        # dimension of the problem
        self.d = d

        # exponent for LinUCB
        self.exp = exp

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

    def apply_poly(self, coef, x):
        ''' Returns the result gotten by applying a poplynomial with given coefficients to
        a vector of pints x'''

        # polynomial degree (+1)
        degree = len(coef)

        # number of x
        N = len(x)

        # store vector of powers of x
        x_pow = np.zeros((N, degree))

        for i in range(degree):
            if i>0:
                x_pow[:,i] = x**(i)
            else:
                x_pow[:,i] = 1
        
        return np.matmul(x_pow, coef.reshape(-1,1)).reshape(-1)
    

    def get_legendre_poly(self, n):
        ''' get the coefficients of the n-th legendre polynomial'''

        coef = np.zeros(n+1)
        for k in range(int(n/2)+1):
            coef[n-2*k] = (-1)**k * 2**(-n) * comb(n, k) * comb (2*n-2*k, n)
        return coef



    def make_legendre_arms(self):
        self.linUCBarms = np.zeros((self.n_arms, self.d))

        # build linear features from the arms
        for j in range(self.d):
            
            # compute degree j legendre polynomial
            coef = self.get_legendre_poly(j)
            
            # apply polynomial to the arms
            self.linUCBarms[:,j] = self.apply_poly(coef, self.arms)


    def make_legendre_even_arms(self):
        self.linUCBarms = np.zeros((self.n_arms, self.d))

        # build linear features from the arms
        for j in range(self.d):
            
            # compute degree 2j legendre polynomial (not j, here we want only even polynomials)
            coef = self.get_legendre_poly(2*j)
            
            # apply polynomial to the arms
            self.linUCBarms[:,j] = self.apply_poly(coef, self.arms)

        
    def make_linUCB(self, lam, m):

        # prepare feature matrix
        if self.even:
            self.make_legendre_even_arms()
        else:
            self.make_legendre_arms()

        # initialize linUCB
        print('Instance linUCB with T='+str(self.T))
        self.learner = linUBC(self.linUCBarms, lam=lam, T=self.T, m=m, exp=self.exp)

    def pull_arm(self):
        
        # ask what arm to pull
        _, arm = self.learner.pull_arm()

        return arm
    
    def update(self, arm, reward):

        # pass from arm index to arm vector
        arm_vector = self.linUCBarms[arm,:]

        # update base learner
        self.learner.update(arm_vector, reward)

