import numpy as np



def I(p,q):
    '''
    Generic function to compute KL divergence between
    bernoulli distributions
    '''
    delta = 0.001
    p = np.clip(p, delta, 1-delta)
    q = np.clip(q, delta, 1-delta)
    if (q == 1) or (q == 0):
        return 100000
    if (p == 1):
        return np.log(p/q)
    if (p == 0):
        return np.log((1-p)/(1-q))
    return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))

def f(n, K):
    '''
    Generic function called in the CKL_UCB algorithm
    '''
    if (n<3):
        return 0
    return np.log(n)+(3*K+1)*np.log(np.log(n))


class CKL_UCB():
    '''
    Algorithm proposed in the article
    'Lipschitz Bandits: Regret Lower Bounds and Optimal Algorithms'
    '''

    def __init__(self, arms, L, tresh = 0.0001):
        '''
        arms = arms of the environment
        L = Lipschitz constant of the reward function
        thresh = parameter involved in the function beta
        '''
        self.arms = arms
        self.N = len(arms)
        self.rewards = np.zeros(self.N)
        self.times_pulled = np.zeros(self.N)
        self.t = 0
        self.leader = 0
        self.L = L
        self.tresh = tresh

    def lam(self, q, k, _k):
        return q-self.L*abs(self.arms[k]-self.arms[_k])

    def summator(self, q, k):
        s = 0
        for _k in range(self.N):
            em_mean = self.rewards[_k]/self.times_pulled[_k]
            s += self.times_pulled[_k]*I(em_mean, self.lam(q, k, _k))
        return s

    def beta(self, arm):
        em_mean = self.rewards[arm]/self.times_pulled[arm]
        delta = (1 - em_mean)/2
        curr = (1 + em_mean)/2
        it = 0
        while (delta>self.tresh):
            it += 1
            if self.summator(curr, arm)<f(self.t, self.N):
                delta /= 2
                curr += delta
            else:
                delta /= 2
                curr -= delta
        return curr

    def choose_arm(self):
        arm = 0
        for arm in range(self.N):
            if(self.times_pulled[arm] < np.log(np.log(self.t + 3))):
                return arm

        self.leader = np.argmax(self.rewards/self.times_pulled)

        beta_curr = self.beta(self.leader)
        possible_arms = []
        for arm in range(self.N):
            if self.summator(beta_curr, arm)<f(self.t, self.N):
                possible_arms.append(arm)
        if (len(possible_arms)>0):
            min_pulled = min(self.times_pulled[possible_arms])
            for arm in possible_arms:
                if self.times_pulled[arm] == min_pulled:
                    return arm
        return self.leader

    def update(self, arm, reward):
        self.t += 1
        self.times_pulled[arm] += 1
        self.rewards[arm] += reward
        if self.t % 1000 == 0:
            print('CKL iteration {}'.format(self.t))

    def reset(self):
        self.rewards = np.zeros(self.N)
        self.times_pulled = np.zeros(self.N)
        self.t = 0


class ZOOM():
    '''
    ZOOMING algorithm, see
    'Multi-Armed Bandits in Metric Spaces'
    '''

    def __init__(self, arms, iph=15, warmup=5):
        '''
        arms = arms of the environment
        iph = exploration parameter corresponding to the order of the time horizon (see the article)
        warmup = inital random arms pulled
        '''
        self.arms = arms
        self.N = len(arms)
        self.warmup = warmup
        self.active_arms = []
        self.active_arms_idx = []
        self.mu = []
        self.n = []
        self.t = 0
        self.iph = iph

    def covered_point(self, p):
        for j in range(len(self.active_arms)):
            radius = (8*self.iph/(2+self.n[j]))**0.5
            if(abs(self.active_arms[j]-p)<radius):
                return True
        return False

    def covering_oracle(self):
        for i in range(self.N):
            if (not self.covered_point(self.arms[i])):
                return self.arms[i], i, False
        return 0, 0, True

    def pull_arm(self):
        if (self.t < self.warmup):
            idx = (self.N*self.t)//self.warmup
            self.active_arms.append(self.arms[idx])
            self.active_arms_idx.append(idx)
            self.n.append(0)
            self.mu.append(0)

            return idx

        arm, idx, guess = self.covering_oracle()
        if not guess:
            self.active_arms.append(arm)
            self.active_arms_idx.append(idx)
            self.n.append(0)
            self.mu.append(0)

        current_best = 0
        amax = 0
        for j in range(len(self.active_arms)):
            radius = (8*self.iph/(2+self.n[j]))**0.5
            if (radius + self.mu[j] > current_best):
                current_best = radius + self.mu[j]
                amax = j

        return self.active_arms_idx[amax]

    def update(self, arm_idx, reward):
        self.t += 1
        for i in range(len(self.active_arms)):
            if (self.active_arms_idx[i] == arm_idx):
                self.mu[i] = (self.n[i]*self.mu[i] + reward)/(self.n[i]+1)
                self.n[i] += 1
                return


    def reset(self):
        self.active_arms = []
        self.active_arms_idx = []
        self.mu = []
        self.n = []
        self.t = 0