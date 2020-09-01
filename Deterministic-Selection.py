import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt

def deterministic_select(current_array, k, m):
    n = len(current_array)
    if n <= m:
        return(np.sort(current_array)[k])
    else:
        small_arrays = [current_array[i:i+m] for i in range(0, n, m)]
        medians = []
        for group in small_arrays:
            if len(group)%2 == 1:
                medians.extend([deterministic_select(group, int(len(group)/2), m)])
            else:
                medians.extend([0.5 * \
                                (deterministic_select(group, int(len(group)/2-1), m)\
                                 + deterministic_select(group, int(len(group)/2), m))])
        #for group in small_arrays:
        #    if len(group)%2 == 1:
        #        medians.extend([np.sort(group)[int(len(group)/2)]])
        #    else:
        #        medians.extend([0.5 * (np.sort(group)[int(len(group)/2)]\
        #                            + np.sort(group)[int(len(group)/2 - 1)])])
        
        #medians = [sorted(group)[int(len(group)/2)] for group in small_arrays]
        
        pivot = deterministic_select(medians, int(len(medians)/2), m)
        
        lower_set = [j for j in current_array if j < pivot]
        higher_set = [j for j in current_array if j > pivot]
        equal_set = [j for j in current_array if j == pivot]
        
        n1 = len(lower_set)
        n2 = len(equal_set)
        
        if k < n1:
            return(deterministic_select(lower_set, k, m))
        elif k >= n1+n2:
            return(deterministic_select(higher_set, k-n1-n2, m))
        else:
            return(pivot)
               
trials = 10
sizes = [5, 7, 9, 11, 13]

fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (8, 8))
axes_list = [item for sublist in axes for item in sublist] 

for n in range(1000, 10000, 1000):
    ax = axes_list.pop(0)
    average = []
    for m in sizes:
        t = []
        for i in range(1, trials+1):
            current_array = [random.randint(1, 100*trials) for _ in range(n)]
            k = int(math.ceil(n/2))
            t1 = time.time()
            deterministic_select(current_array, k, m)
            t2 = time.time()
            t.extend([t2-t1])
        average.append(np.mean(t))
    ax.plot(sizes, average)
    ax.set_title('Array Size = {}'.format(n))

plt.tight_layout()