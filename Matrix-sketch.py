#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:45:23 2020

@author: Ashley Shang 
"""

import sys
import os
import random
import numpy as np 


def matrix_sketch(mat, eps):
    cols = int(np.ceil(2/eps))
    rows = len(mat)
    if cols>rows:
        cols = rows
    sketch_trans = [[0 for _ in range(rows)] for _ in range(cols)]
    for col in np.array(mat).transpose():
        if sum(np.array(sketch_trans).transpose().any(0)) != cols:
            l = np.array(sketch_trans).transpose().any(0)
            ind = [i for i, val in enumerate(l) if ~val] 
            k = random.sample(ind, 1)[0]
            sketch_trans[k] = col
        if sum(np.array(sketch_trans).transpose().any(0)) == cols:
            sketch = np.array(sketch_trans).transpose()
            u, sigma, v = np.linalg.svd(sketch)
            theta = sigma[int(np.ceil(cols/2)-1)]**2
            sigma_square_diff = [sigma[i]**2 for i in range(len(sigma))] - theta
            sigma_square_diff_revise = [x if x>0 else 0 for x in sigma_square_diff]
            sigma_star = [np.sqrt(x) for x in sigma_square_diff_revise]
            sketch = np.dot(u[:,:cols], np.diag(sigma_star))
            sketch_trans = sketch.transpose()
    return sketch_trans.transpose()

if __name__ == '__main__':
    ori_rows = int(sys.argv[1])
    ori_cols = int(sys.argv[2])
    eps = float(sys.argv[3])
    file = sys.argv[4]
    output = sys.argv[5]
    with open(file) as f:
        data = [[float(num) for num in line.split(' ') if num != '\n'] for line in f]
    
    data = np.matrix(data)
    matrix = np.dot(data,data.transpose())
    
    print('-------------------------------------------------')
    print('Original Data-Matrix has {0}-rows & {1}-columns'.format(ori_rows,ori_cols))
    print('Epsilon = {0}, (i.e. max. of {1}% reduction of Frobenius-Norm of the Sketch Matrix)'.format(eps, int(eps*100)))
    print('Input File = {0}'.format(file))
    print('Frobenius-Norm of the ({0} x {1}) Data Matrix = {2}'.format(ori_rows,ori_cols,round(np.linalg.norm(matrix),1)))
    
    b = matrix_sketch(data, eps)
    sketch = np.dot(b, b.transpose())
    
    
    print('Frobenius-Norm of the ({0} x {1}) Data Matrix = {2}'\
      .format(b.shape[0],b.shape[1],round(np.linalg.norm(sketch),1)))
    print('Change in Frobenius-Norm between Sketch & Original = {}%'\
      .format((np.linalg.norm(sketch) - np.linalg.norm(matrix))/np.linalg.norm(matrix)*100))
    print('File \'{0}\' contains a ({1} x {2}) Matrix-Sketch'.format(output, b.shape[0], b.shape[1]))

    mat = np.matrix(b)
    if os.path.exists(output):
        os.remove(output)
    with open(output,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.5f')







    
    