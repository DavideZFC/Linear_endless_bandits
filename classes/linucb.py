import numpy as np

class linUBC:
    def __init__(self, arms_matrix, lam = 1):
        self.d = arms_matrix.shape[1]
        self.arms = arms_matrix
        self.design_matrix = lam*np.identity(self.d)
        self.load = np.zeros(self.d)

    def update(self, arm, reward):
        self.design_matrix += np.matmul(arm.reshape(-1,1),arm.reshape(1,-1))
        self.load += reward*arm.reshape(-1,1)

    def pull(self):
        return np.linalg.solve(self.design_matrix, self.load)



    

