from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ConstantKernel as C
import numpy as np

class Gauss_Bandit():
    '''
    a.k.a. Gaussian UCB1 is very good in practice but has not the same theoretical guarantees of the others, 
    and so it is widely not considered a valid baseline
    '''
    def __init__(self, arms, update_every=50):
        '''
        arms = arms of the environment
        '''
        self.arms = arms
        self.N = len(arms)
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.step = 0
        self.update_every = update_every

    def pull_arm(self):
        self.step += 1
        if self.step < self.N:
            return self.step
        else:
            mean, std = self.gp.predict(self.arms.reshape(-1,1), return_std = True)
            return np.argmax(mean + std)

    def update(self, arm, reward):
        self.eval_x.append(self.arms[arm])
        self.eval_y.append(reward)
        if (self.step > 10 and self.step % self.update_every == self.update_every-1):
            self.gp.fit(np.array(self.eval_x).reshape(-1, 1), np.array(self.eval_y))

    def reset(self):     
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.step = 0



class GPTS():
    '''
    Gaussian Thompson Sampling is very good in practice but has not the same theoretical guarantees of the others, 
    and so it is widely not considered a valid baseline
    '''
    
    def __init__(self, arms, update_every=50):
        '''
        arms = arms of the environment
        update_every = how frequently to change the distribution of the gp process (divides the computational time without loss in performance)
        '''
        self.arms = arms
        self.N = len(arms)
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.step = 0
        self.means = np.zeros(self.N)
        self.stds = np.zeros(self.N)
        self.update_every = update_every

    def pull_arm(self):
        self.step += 1
        if (self.step % 1000 == 0):
            print('GPTS siamo allo step = '+str(self.step))
        if self.step < self.N:
            return self.step
        else:
            if(self.step % self.update_every == 0):
                self.means, self.stds = self.gp.predict(self.arms.reshape(-1,1), return_std = True)
            samples = self.stds*np.random.randn(self.N)+self.means
            return np.argmax(samples)

    def update(self, arm, reward):
        self.eval_x.append(self.arms[arm])
        self.eval_y.append(reward)
        if (self.step > 10 and self.step % self.update_every==self.update_every-1):
            self.gp.fit(np.array(self.eval_x).reshape(-1, 1), np.array(self.eval_y))

    def reset(self):     
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.step = 0
        self.means = np.zeros(self.N)
        self.stds = np.zeros(self.N)


    

