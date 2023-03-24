import numpy as np


def diaginv(A):
    d = A.shape[0]
    B = np.zeros_like(A)
    for i in range(d):
        B[i,i] = A[i,i]**(-1)
    return B

def makediag(D):
    I = np.identity(d)
    for i in range(d):
        I[i,i] = D[i]
    return I



def find_maximum(A, v):
    D, U = np.linalg.eig(A)
    
    D = makediag(D)
    invD = diaginv(D)


    matrix = np.matmul(invD**(0.5),U.T)
    v_based = np.matmul(matrix, v.reshape(-1,1))
    
    return (np.sum(v_based**2))**0.5




d = 2
A0 = np.random.normal(size=(d,d))

A = np.matmul(A0.T,A0)
A = np.array([[1.001, 1], [1, 1.001]])
print(A)
D, U = np.linalg.eig(A)

I = np.identity(d)
for i in range(d):
    I[i,i] = D[i]

A_rec = np.matmul(np.matmul(U, I),U.T)



v = np.ones(d)
v[1] = -1
print(find_maximum(A,v))