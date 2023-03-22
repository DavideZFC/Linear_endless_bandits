import numpy as np
import matplotlib.pyplot as plt

# time horizon
T = 100000

# smoothness of the function
p = 9

# compute optimal d
alpha = 1/(2*p)
d = T**(alpha)

# compute number of arms
k = int(T**0.5)

print('Time horizon: {}'.format(T))
print('Dimension: {}'.format(d))
print('Number of arms: {}'.format(k))

a = -1
b = 1
x = np.linspace(a,b,k)

def make_feature_matrix(x, d):
    # serie di soli seni
    k = len(x)
    phi = np.zeros((k,d))
    for i in range(k):
        for j in range(d):
            phi[i,j] = np.sin((j+1)*np.pi*x[i])
    return phi

d = 10
phi = make_feature_matrix(x,d)

plt.scatter(phi[:,0], phi[:,1])
# for i in range(d):
#    plt.plot(x,phi[:,i])
plt.show()