import numpy as np
import matplotlib.pyplot as plt
from classes.legendreucb import LegendreUCB
from classes.chebishevucb import ChebishevUCB
from classes.fourierucb import FourierUCB
from classes.meta_learner import MetaLearner
from classes.bases import *

for d in range(5, 2):
    mat = np.zeros((d,d))
    N = 1000
    x = np.linspace(-1,1,N)

    for i in range(d):
        v = get_legendre_norm_poly(i)
        mat[i,:len(v)] = v

    mat = mat.T

    print('Real norm for d={}'.format(d))
    print(np.linalg.norm(mat, ord=2))

'''
_, _, VT = np.linalg.svd(mat)

u = VT[0,:].T
print(np.linalg.norm(u))

L = make_legendre_norm_arms(N, d, x)

y = np.dot(L, u)
plt.plot(x,y)
plt.show()

# same polynomial in the two bases
coef_ = np.dot(mat,u)
y = apply_poly(coef_, x)

print(coef_)

plt.plot(x,y)
plt.show()
'''