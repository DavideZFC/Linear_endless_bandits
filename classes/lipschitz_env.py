import numpy as np
import matplotlib.pyplot as plt
from functions.critical_poly import critical_poly

class Lipschitz_Environment:
    '''
    benchmark environments.
    The rewards are always distributed according to Bernoulli, and the reward function is given by y.
    The arms are self.x, they lay in the interval [0,lim]
    '''

    def __init__(self, n_arms=200, lim=1, sigma=0.1, curve='sin-like', seed = 257):
        '''
        numel = number of arms
        lim = range of the arms [0,lim]
        L = lipschitz function to be granted
        curve = sin-like/spike/random is the benchmark we want to use (see the images)
        spike_par = see construction of the curve
        seed = random seed to be used
        '''

        np.random.seed(seed)

        # name of the curve we want
        self.curve = curve

        # set of arms
        self.x = np.linspace(-lim, lim, n_arms)
        self.lim = lim

        # number of arms
        self.n_arms = n_arms

        # reward curve
        self.y = np.zeros(n_arms)

        # difference between consecutive arms
        self.h = self.x[1] - self.x[0]

        # standard deviation of the noise
        self.sigma = sigma

        # make curve
        self.generate_curves(curve)

        # compute optimum
        self.get_optimum()
        
            

    def generate_curves(self, curve):

        # Cinfty even
        if (curve == 'gaussian'):
            sigma = 0.3
            self.y = (1/2*np.pi*sigma**2)**0.5*np.exp(-(self.x)**2/(2*sigma**2))

        if (curve == 'bigaussian'):
            sigma = 0.3
            self.y = 2.*(1/2*np.pi*sigma**2)**0.5*np.exp(-(self.x)**2/(2*sigma**2))

        # even sinusoidal
        elif (curve == 'cosine'):
            self.y = np.cos(np.pi*self.x)

        # polynomial even
        elif (curve == 'even_poly'):
            z1 = -0.75
            z2 = 0.75
            self.y = -(self.x - z1)**2*(self.x - z2)**2

        elif (curve == 'bigeven_poly'):
            z1 = -0.75
            z2 = 0.75
            self.y = -3.*(self.x - z1)**2*(self.x - z2)**2

        # generic polynomial
        elif (curve == 'poly'):
            a = -1
            b = 1
            c = -1
            self.y = a*self.x**2 + b*self.x + c

        # multimodal not even function
        elif (curve == 'sin-like'):
            for i in range(self.n_arms):
                self.y[i] = ((self.x[i]-0.2)/self.lim)*np.sin(5*self.x[i])

        # very ugly even function
        elif (curve == 'spike'):
            spike_par = 0.5
            mean_x = self.x[self.n_arms//2]
            for i in range(self.n_arms):
                if (abs(self.x[i]-mean_x)<spike_par):
                    self.y[i] = spike_par - abs(self.x[i]-mean_x)

        elif (curve == 'bigspike'):
            spike_par = 0.5
            mean_x = self.x[self.n_arms//2]
            for i in range(self.n_arms):
                if (abs(self.x[i]-mean_x)<spike_par):
                    self.y[i] = 2.*(spike_par - abs(self.x[i]-mean_x))

        # ugly and not symmetric
        elif (curve == 'random'):
            for i in range(1,self.n_arms):
                self.y[i] = self.y[i-1] + 10*np.random.uniform(-self.h, self.h)

        elif (curve == 'symrandom'):
            for i in range(1,int(self.n_arms//2)+1):
                self.y[i] = np.clip(self.y[i-1] + np.random.uniform(-self.h, self.h), 0,1)
                self.y[self.n_arms - i] = self.y[i]


        elif (curve == 'francesco'):
            spike_par = 0.3
            mean_x = self.x[self.n_arms//2]
            eta = 0.65
            for i in range(self.n_arms):
                if (abs(self.x[i]-mean_x)<spike_par):
                    self.y[i] = spike_par - abs(self.x[i]-mean_x)
                elif (abs(self.x[i]-mean_x)<(1+eta)*spike_par and self.x[i]>mean_x):
                    self.y[i] = abs(spike_par - abs(self.x[i]-mean_x))
                elif(self.x[i] > mean_x):
                    self.y[i] = eta*spike_par
        
        elif curve == 'critical_poly':
            self.y = critical_poly(self.x)

        elif curve == 'expert':
            self.y = (self.x+1)*np.exp(1/(self.x**2-1.0001))
        
        elif curve == 'expertsin':
            self.y = 3*np.sin(2*np.pi*self.x)*np.exp(1/(self.x**2-1.0001))

        self.plot_curve()


    def pull_arm(self, arm):
        reward = np.random.normal(self.y[arm], self.sigma)
        expected_regret = self.opt - self.y[arm]
        return reward, expected_regret

    def get_optimum(self):
        self.opt = max(self.y)
        return max(self.y)

    def plot_curve(self):
        plt.plot(self.x, self.y)
        plt.xlabel('Arms')
        plt.ylabel('Reward')
        plt.title('Reward curve for {}'.format(self.curve))
        plt.show()
