#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:54:02 2022

@author: palinakroyer
"""


import numpy as np
import matplotlib.pyplot as plt


#data_ref = np.loadtxt("Data/Jacobi_naive/jacobi_naive.dat",unpack=True)
data_naive = np.loadtxt("Data/Jacobi_naive/jacobi_naive.dat",unpack=True)
data_ref = np.loadtxt("Data/Jacobi_ref/jacobi_ref2.dat",unpack=True)


N_ref = data_ref[3]
N_naive = data_naive[3]


flops_ref = data_ref[5]
time_ref = data_ref[2]
iterations_ref = data_ref[4]

flops_naive = data_naive[5]
time_naive = data_naive[2]
iterations_naive = data_naive[4]


N_ref = data_ref[3]
flops_ref = data_ref[5]
time_ref = data_ref[2]
iterations_ref = data_ref[4]



plt.figure(1)
#plt.plot(N_ref,flops_ref,'*-')
plt.plot(N_naive,flops_naive/1000,'*-')
plt.plot(N_ref,flops_ref/1000,'*-')
plt.title("")
plt.xlabel("N, grid size")
plt.ylabel("Performance [Gflops/s]")
#plt.ylim(0,50)
plt.legend(["Naive, GPU", "Reference, 12 thread"]) 
    
plt.show()


    


