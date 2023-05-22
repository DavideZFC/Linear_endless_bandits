import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

class GPR:
    def __init__(self, kernel, sigma_y=0.01):
        self.kernel = kernel
        self.sigma_y = sigma_y

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.K = self.kernel(X, X)
        self.K += self.sigma_y**2 * np.eye(X.shape[0])
        self.L = np.linalg.cholesky(self.K)

    def predict(self, X_test, return_std=True):
        K_s = self.kernel(self.X_train, X_test)
        L_inv = np.linalg.solve(self.L, K_s)
        mu = np.dot(L_inv.T, np.linalg.solve(self.L, self.y_train))
        var = self.kernel(X_test, X_test) - np.dot(L_inv.T, L_inv) + self.sigma_y**2
        if return_std:
            return mu.flatten(), np.diagonal(var)
        else:
            return mu.flatten()
    
def werner(x1, x2, n=8):
    # epsilon is added to the diagonal to make it less singular

    d1 = x1.shape[0]
    d2 = x2.shape[0]

    mat = np.zeros((d1,d2))
    for i in range(d1):
        for j in range(d2):
            if x1[i,0]-x2[j,0]==0:
                mat[i,j] = n
            else:
                delta = x1[i,0]-x2[j,0]
                mat[i,j] = np.sin(n*delta/2)/np.sin(delta/2)*np.cos((n-1)*delta/2)

    
    return mat



class IGPUCB(object):

    def __init__(self, arms, T=10000, B=4, R=1, update_every=10, kernel='gaussian'):

        self.arms = arms
        self.T = T
        self.delta = 1/T
        self.B = B
        self.R = R
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
        return GPR(kernel=werner)


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
        gamma = np.log(self.t)**2
        beta = self.B + self.R*np.sqrt(2*gamma + 1 + np.log(1/self.delta))

        return np.argmax(self.mu + self.sigma * np.sqrt(beta))
  
    def update(self, arm, reward):
        self.X.append(self.arms[arm])
        self.y.append(reward)


