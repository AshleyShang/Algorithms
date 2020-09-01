#!/usr/bin/env python
# coding: utf-8

import sys
import math
from collections import deque

# set up the tower
tower = deque()

number_of_steps = 0

# set the max recursion times
sys.setrecursionlimit(5000)

# build up the pegs
def initialize(n, k) :
    for i in range(k):
        peg = deque()
        if i == 0:
            for j in range(n):
                peg.append(j+1)
        tower.append(peg)
        
# check the legality
def check_legal():
    flag = True
    for i in range(len(tower)):
        for j in range(len(tower[i])):
            for k in range(j, len(tower[i])):
                if tower[i][j] > tower[i][k]:
                    flag = False
    return(flag)

# define the function that move the bottom disk                    
def move_the_last(source, dest) :
    global number_of_steps
    number_of_steps += 1
    disk = tower[source].popleft()
    tower[dest].appendleft(disk)
    if check_legal() == True:
        y = " (legal)"
    else:
        y = " (ilegal)"    
    print("Move disk "+str(disk)+" from peg "+str(source+1)+" to peg "+str(dest+1)+y)


# define the function that is for 3-peg hanoi
def three_pegs_move(n, source, dest, intermediate):
    if n == 1:
        move_the_last(source, dest)
    else:
        three_pegs_move(n-1, source, intermediate, dest)
        move_the_last(source, dest)
        three_pegs_move(n-1, intermediate, dest, source)
        
# now for k pegs
# for given source and destination, select the second peg as 
# intermediate1, move the first n-p disks to the second peg.
# then pick one of the rest pegs as intermediate2, trans the
# question to 3-peg hanoi problem and move the rest p disks to
# the destination.
# At last, move the n-p disks that on inter1 to destination
def k_pegs_move(n, source, dest, inter, *rest):
    if n > 0 :
        p = math.floor(math.sqrt(2*n))
        rest = list(rest)
        inter2 = rest.pop()
        rest.extend([dest])
        k_pegs_move(n-p, source, inter, inter2, *rest)
        rest.remove(dest)
        three_pegs_move(p, source, dest, inter2)
        rest.extend([source])
        k_pegs_move(n-p, inter, dest, inter2, *rest)
        rest.remove(source)
        
def print_peg_state(peg):
    global number_of_steps
    print ("-------------------------------")
    print("The state of peg {0} (top to bottom): {1}".format(str(peg+1), tower[peg]))
    print("Number of steps: {}".format(number_of_steps))
    print ("-------------------------------")
    
# read the input and call the function
if __name__ == '__main__' :
    n = int(sys.argv[1])
    k = int(sys.argv[2])
    initialize(n, k)
    print_peg_state(0)
    k_pegs_move(n, 0, k-1, 1, *list(range(2, k-1)))
    print_peg_state(k-1)