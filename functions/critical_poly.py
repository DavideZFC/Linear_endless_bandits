import numpy as np
from classes.bases import *


def critical_poly(x ,d=9):
    N = len(x)
    mat = np.zeros((d,d))

    for i in range(d):
        v = get_legendre_norm_poly(i)
        mat[i,:len(v)] = v

    mat = mat.T

    _, _, VT = np.linalg.svd(mat)

    u = VT[0,:].T
    print(np.linalg.norm(u))

    L = make_legendre_norm_arms(N, d, x)

    y = np.dot(L, u)

    # same polynomial in the two bases
    coef_ = np.dot(mat,u)

    y = apply_poly(coef_, x)
    return -y
