import numpy as np

class curve_fit:
    '''
    Class to make curve fit
    '''

    def __init__(self, sigma=0.5, curve='gaussian', seed = 257):
        '''
        Defines the variables in the environment

        Parameters:
        sigma (double): standard deviation of the noise
        curve (str): name of the reward curve to build. Possible values:
            'gaussian',
        seed (int): random seed to fix
        '''

        np.random.seed(seed)

        # standard deviation of the noise
        self.sigma = sigma

        # make curve
        self.generate_curves(curve)
        
            

    def generate_curves(self, curve):
        '''
        Generate the reward curve corresponing to its name

        Parameters:
        curve (str): name of the curve
        '''

        if curve == 'gaussian':
            def curve(x):
                a = 2.0
                b = 1.0
                return a*np.exp(-x**2)-b
        if curve == 'jump':
            def curve(x):
                a = 1.0
                b = 0.0
                return 1*(x>0)
        if curve == 'sin':
            def curve(x):
                omega = 2
                return np.sin(np.pi*omega*x)
        if curve == 'cos':
            def curve(x):
                omega = 1
                return np.cos(np.pi*omega*x)
        if curve == 'periodic':
            def curve(x):
                omega = 1
                s = np.sin(np.pi*omega*x)
                c = np.cos(np.pi*omega*x)
                return np.exp(2*s)+c**2
        if curve == 'sigmoid':
            def curve(x):
                a = 10
                return 1/(np.exp(-a*x)+1)
        if curve == 'custom':
            def curve(x):
                a = 10
                c = 2.0
                b = 1.0
                return c*np.exp(-(x+0.5)**2)-b+1/(np.exp(-a*x)+1)
        
        self.curve = curve
    
    def get_sample(self, x):
        '''
        Get a sample from the curve
        '''
        noise = np.random.normal(0,self.sigma)
        return self.curve(x) + noise