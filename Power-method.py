# -*- coding: utf-8 -*-

import numpy as np

A = [[1, 2], [3,4]]
A = np.matrix(A)

A_t = A.transpose()
B = A_t.dot(A)

r = np.linalg.matrix_rank(A)

k = 10

V = []
S = []
U = []

def matrix_multiplication(M, k):
    if k == 1:
        return M
    else:
        if k % 2 == 0:
            return matrix_multiplication(M*M, int(k/2))
        else:
            return M * matrix_multiplication(M*M, int((k-1)/2))


def get_svd(m, k, r):
    mat = matrix_multiplication(m, k)
    v = mat[:,0]/np.linalg.norm(mat[:,0])
    sigma = np.linalg.norm(A*v)
    u = A*v/sigma
    
    V.append(v.flatten().tolist()[0])
    S.append(sigma)
    U.append(u.flatten().tolist()[0])
    
    
    if r == 1:
        return V, S, U
    else:
        return get_svd(m - sigma**2*v*v.transpose(), k, r-1)
    

np.set_printoptions(precision=5)
v, s, u = get_svd(B, k, r)
v = np.array(v)
s = np.array(s)
u = np.array(u)
    
    