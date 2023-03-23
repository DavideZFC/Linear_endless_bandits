import numpy as np

class linUBC:
    def __init__(self, arms_matrix, lam=1, T=1000000, m=1):

        # dimension of the arms
        self.d = arms_matrix.shape[1]

        # lambda parameter of ridge regression
        self.lam = lam

        # matrix of the arms
        self.arms = arms_matrix

        # initialization of the matrix A of the system
        self.design_matrix = lam*np.identity(self.d)

        # initialization of the known term b of the system
        self.load = np.zeros(self.d).reshape(-1,1)

        # time horizon
        self.T = T

        # error probability
        self.delta = 1/T

        # upper bound on the norm of theta
        self.m = m

        # maximum norm of an arm vector
        self.L = 0
        for i in range(self.arms.shape[0]):
            self.L = max(self.L,self.norm(self.arms[i,:]))
        
        print('Highest norm estimated {}'.format(self.L))

        self.compute_beta_routine()

    def norm(self,v):
        return (np.sum(v**2))**0.5

    def compute_beta_routine(self):
        self.beta = np.zeros(self.T)

        for t in range(self.T):
            first = self.m * self.lam**0.5
            second = 2*np.log(1/self.delta) + self.d*np.log((self.d*self.lam+t*self.L**2)/(self.d*self.lam))
            second = second**0.5
            self.beta[t] = first + second

    def update(self, arm, reward):
        self.design_matrix += np.matmul(arm.reshape(-1,1),arm.reshape(1,-1))
        self.load += reward*arm.reshape(-1,1)

    def estimate_theta(self):
        return np.linalg.solve(self.design_matrix, self.load)



    

