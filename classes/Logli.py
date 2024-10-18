
import numpy as np

class Logli():
    '''
    ZOOMING algorithm, see
    'Multi-Armed Bandits in Metric Spaces'
    '''

    def __init__(self, env, T, B=10):
        '''
        arms = arms of the environment
        iph = exploration parameter corresponding to the order of the time horizon (see the article)
        warmup = inital random arms pulled
        '''
        self.env = env
        self.T = T
        self.reward_story = np.zeros(T)
        self.C = hypercube(0,1)
        self.B = B
        self.r = 2.0**(-np.arange(self.B)-1)
        self.n = (16*np.log(T)/(self.r**2)).astype(int)
        self.t = 0
        self.mu = [0]
        self.m = 0


    def execute(self):
        self.roundfunc(self.m, self.C, 0)

    def roundfunc(self, m, C, h):
        if self.t+self.n[h] > self.T:
            return
        else:
            x_vec = C.sample(self.n[h])
            y_vec = np.zeros(self.n[h])
            for i in range(self.n[h]):
                y_vec[i] = self.env.get_sample(x_vec[i])
                self.t += 1
                self.reward_story[self.t] = y_vec[i]
            y_mean = np.mean(y_vec)
            if h == m:
                self.mu[m] = max(self.mu[m], y_mean)
            else:
                if self.mu[m-1] - y_mean < 4*self.r[h]:
                    cubes = C.split(self.r[h+1]/self.r[h])
                    for c in cubes:
                        self.roundfunc(m, c, h+1)
                else:
                    return

class hypercube:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def sample(self, n):
        return np.random.uniform(self.a, self.b, n)
    
    def split(self, factor):
        l = (self.b - self.a)*factor
        new_cubes = []
        x = self.a
        while x < self.b:
            new_cubes.add(hypercube(x,x+l))
            x = x+l