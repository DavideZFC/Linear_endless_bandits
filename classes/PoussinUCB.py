from classes.linucb import linUBC
from classes.misspec import misSpec
from classes.PE import PE
import numpy as np
from classes.bases import *



class PoussinUCB:
    def __init__(self, arms, d, lam=1, T=10000, m=1, pe=False, epsilon=0, IPERPARAMETRO=1, n_pous=10):
        # dimension of the problem
        self.d = d

        # arms avaiable
        self.arms = arms
        self.n_arms = len(self.arms)

        # time horizon
        self.T = T
        self.t = 0

        # poussin kernel to use
        self.n_pous = n_pous
  
        self.linarms = make_sincos_arms(self.n_arms, d, arms)

        # initialize learner
        if pe:
            self.make_PE(IPERPARAMETRO=IPERPARAMETRO)
        else:
            self.make_linUCB(lam, m)

        self.sample_all_noises()

    def reset(self):
        self.t = 0
        self.learner.reset()
        self.sample_all_noises()

    def make_linUCB(self, lam, m):
        # initialize linUCB
        self.learner = linUBC(self.linarms, lam=lam, T=self.T, m=m)

    def make_PE(self,IPERPARAMETRO):
        # initialize linUCB
        self.learner = PE(self.linarms, T=self.T,IPERPARAMETRO=IPERPARAMETRO)


    def pull_arm(self):
        # ask what arm to pull
        try:
            _, arm = self.learner.pull_arm()
        except:
            arm = self.learner.pull_arm()

        arm_prenoise = self.arms[arm]
        # add noise
        arm_postnoise = self.adjust(arm_prenoise+self.noises[self.t])
        # find nearest index
        self.last_pulled_arm = self.find_nearest_index(arm_postnoise)
        return self.last_pulled_arm
    
    def update(self, arm, reward):
        # pass from arm index to arm vector
        arm = self.last_pulled_arm
        arm_vector = self.linarms[arm,:]
        # update base learner
        self.learner.update(arm_vector, reward*self.sig[self.t])
        # update step
        self.t += 1

    def sample_all_noises(self):

        def PoussinWrapper(n,p):
            c1 = (2*n+1-p)/2
            c2 = (p+1)/2
            c3 = 2*(p+1)
            def Poussin(x):
                return np.sin(np.pi*(c1*x))*np.sin(np.pi*(c2*x))/(c3*np.sin(np.pi*(x/2))**2)
            return Poussin

        def PoussinAbsWrapper(n,p):
            c1 = (2*n+1-p)/2
            c2 = (p+1)/2
            c3 = 2*(p+1)
            def Poussin(x):
                return np.abs(np.sin(np.pi*(c1*x))*np.sin(np.pi*(c2*x))/(c3*np.sin(np.pi*(x/2))**2))
            return Poussin

        f = PoussinAbsWrapper(self.n_pous,self.n_pous//2)
        g = PoussinWrapper(self.n_pous,self.n_pous//2)


        self.noises = np.zeros(self.T)
        self.sig = np.zeros_like(self.noises)
        self.noises[0] = np.random.uniform(-0.1,0.1)

        for i in range(self.T-1):
            x1 = np.random.uniform(-1,1)
            q = f(x1)/f(self.noises[i])
            if np.random.binomial(1,min(q,1)) == 1:
                self.noises[i+1] = x1
            else:
                self.noises[i+1] = self.noises[i]

        for i in range(self.T):
            self.sig[i] = 2*(g(self.noises[i])>0)-1

    def adjust(self, x):
        if x > 1:
            return x-2
        if x < -1:
            return x+2
        return x
    
    def find_nearest_index(self, x):
        return round((x+1)*(self.n_arms-1)/2)
            

        

    


