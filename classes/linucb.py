import numpy as np

class linUBC:
    def __init__(self, arms_matrix, lam=1, T=1000000, m=1):

        # dimension of the arms
        self.d = arms_matrix.shape[1]

        # lambda parameter of ridge regression
        self.lam = lam

        # matrix of the arms
        self.arms = arms_matrix
        self.n_arms = self.arms.shape[0]

        # initialization of the matrix A of the system
        self.design_matrix = lam*np.identity(self.d)

        # initialization of the known term b of the system
        self.load = np.zeros(self.d).reshape(-1,1)

        # time horizon
        self.T = T
        self.t = 0

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

    def reset(self):
        self.design_matrix = self.lam*np.identity(self.d)
        self.load = np.zeros(self.d).reshape(-1,1)
        self.t = 0


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

    def find_maximum(self, A, v):
        ''' computes the maximum of the linear form (x.T*v)(x.T*A*x) '''

        def diaginv(A):
            d = A.shape[0]
            B = np.zeros_like(A)
            for i in range(d):
                B[i,i] = A[i,i]**(-1)
            return B

        def makediag(D):
            d = len(D)
            I = np.identity(d)
            for i in range(d):
                I[i,i] = D[i]
            return I
        D, U = np.linalg.eig(A)
        
        D = makediag(D)
        invD = diaginv(D)


        matrix = np.matmul(invD**(0.5),U.T)
        v_based = np.matmul(matrix, v.reshape(-1,1))
        
        return (np.sum(v_based**2))**0.5
    


    def pull_arm(self):
        estimates = np.zeros(self.n_arms)
        thetahat = self.estimate_theta().flatten()


        for i in range(self.n_arms):
            estimates[i] = np.dot(thetahat, self.arms[i,:])
            upper_bound = self.find_maximum(self.design_matrix, self.arms[i])
            estimates[i] += self.beta[self.t]*upper_bound

        self.t += 1

        return self.arms[np.argmax(estimates)], np.argmax(estimates)



    def estimate_theta(self):
        return np.linalg.solve(self.design_matrix, self.load)



    

