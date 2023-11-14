import numpy as np

class dummy_learner:
    def __init__(self, arms, epsilon, delta):
        self.arms = arms
        self.n_arms = arms.shape[0]
        self.d = arms.shape[1]
        pi = self.compute_optimal_design(arms)

        self.times_to_pull = np.zeros(self.n_arms)
        self.fixed_design = []
        for i in range(self.n_arms):
            term1 = 2*pi(i)*self.d / epsilon**2
            term2 = np.log(self.n_arms/delta)
            self.times_to_pull[i] = np.ceil(term1*term2)
            if self.times_to_pull[i] > 0:
                self.fixed_design.append([i]*self.times_to_pull[i])

        self.idx = 0

        self.design_matrix = np.zeros((self.d, self.d))
        self.load = np.zeros(self.d).reshape(-1,1)

    def update(self, arm, reward):
        self.design_matrix += np.matmul(arm.reshape(-1,1),arm.reshape(1,-1))
        self.load += reward*arm.reshape(-1,1) 

    def get_theta(self):
        return np.solve(self.design_matrix, self.load)  

    def check_status(self):
        return self.idx < len(self.fixed_design) 

    def pull_arm(self):
        self.idx += 1
        return self.arms[self.idx-1]
            
    def compute_optimal_design(A):
        pass




class PE:
    def __init__(self, arms_matrix, T=10000.0):

        # dimension of the arms
        self.d = arms_matrix.shape[1]

        # matrix of the arms
        self.arms = arms_matrix
        self.n_arms = self.arms.shape[0]
        # time horizon
        self.T = T
        self.t = 0

        # error probability
        self.delta = T**(-1)

        # epsilon value
        self.epsilon = 0.1

        # initialize_learner
        self.learner = dummy_learner(arms_matrix, self.epsilon, self.delta)

    def reset(self):
        self.t = 0
        self.epsilon = 0.1

        # initialize_learner
        self.learner = dummy_learner(self.arms, self.epsilon, self.delta)

    def compute_active_arms(self):
        theta = self.learner.get_theta()
        scalar_vector = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            scalar_vector[i] = np.dot(theta, self.arms[i,:])
        
        best = np.max(scalar_vector)
        self.active_arms = scalar_vector > best - 2*self.epsilon

    def update(self, arm, reward):
        self.learner.update(arm, reward)

    def pull_arm(self):
        if self.learner.check_status():
            return self.learner.pull_arm()
        else:
            self.compute_active_arms()
            self.epsilon /= 2
            self.learner = dummy_learner(self.arms[self.active_arms], self.epsilon, self.delta)
            self.learner.pull_arm()






    

