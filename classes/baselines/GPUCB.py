import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel


class GPUCB(object):

    def __init__(self, arms, beta=100., update_every=10, kernel='gaussian'):
        '''
        meshgrid: Output from np.methgrid.
        e.g. np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1)) for 2D space
        with |x_i| < 1 constraint.
        environment: Environment class which is equipped with sample() method to
        return observed value.
        beta (optional): Hyper-parameter to tune the exploration-exploitation
        balance. If beta is large, it emphasizes the variance of the unexplored
        solution solution (i.e. larger curiosity)
        '''
        self.arms = arms
        self.beta = beta
        self.t = 0
        self.kernel = kernel
        if self.kernel == 'gaussian':
            print('Build gaussian kernel')
            self.gp = GaussianProcessRegressor(normalize_y=True)
        elif self.kernel == 'dirichlet':
            print('Build dirichlet kernel')
            self.gp = self.make_dirichlet_kernel()

        self.mu = np.zeros(len(self.arms))
        self.sigma = np.ones(len(self.arms))
        self.update_every = update_every
        self.X = []
        self.y = []

    def make_dirichlet_kernel(self):
        d = 8
        def dirichlet(x,y):
            return np.sin((d+0.5)*(x-y))/np.sin((x-y)/2)
        k = Kernel(dirichlet)
        return GaussianProcessRegressor(normalize_y=True, kernel=k)


    def pull_arm(self):
        self.t += 1
        if self.t % self.update_every == 0:
            self.gp.fit(np.array(self.X).reshape(-1, 1), np.array(self.y))
            self.mu, self.sigma = self.gp.predict(self.arms.reshape(-1,1), return_std = True)
        return self.argmax_ucb()
    
    def reset(self):
        self.t = 0
        self.mu = np.zeros(len(self.arms))
        self.sigma = np.ones(len(self.arms))
        if self.kernel == 'gaussian':
            self.gp = GaussianProcessRegressor(normalize_y=True)
        elif self.kernel == 'dirichlet':
            self.gp = self.make_dirichlet_kernel()
        self.X = []
        self.y = []


    def argmax_ucb(self):
        return np.argmax(self.mu + self.sigma * np.sqrt(self.beta))
  
    def update(self, arm, reward):
        self.X.append(self.arms[arm])
        self.y.append(reward)


