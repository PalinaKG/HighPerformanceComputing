#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:54:02 2022

@author: palinakroyer
"""


import numpy as np
import matplotlib.pyplot as plt


#data_ref = np.loadtxt("Data/Jacobi_naive/jacobi_naive.dat",unpack=True)
data_multi = np.loadtxt("Data/Jacobi_multi/jacobi_multi.dat",unpack=True)
data_ref = np.loadtxt("Data/Jacobi_ref/jacobi_ref3.dat",unpack=True)
data_naive = np.loadtxt("Data/Jacobi_naive/jacobi_naive.dat",unpack=True)
data_seq = np.loadtxt("Data/Jacobi_seq/jacobi_seq.dat",unpack=True)
data_optnaive = np.loadtxt("Data/Jacobi_naive/jacobi_optnaive.dat",unpack=True)


N_ref = data_ref[3]
N_multi = data_multi[3]
N_naive = data_naive[3]
N_seq = data_seq[3]
N_optnaive = data_optnaive[3]


flops_ref = data_ref[5]
time_ref = data_ref[2]
iterations_ref = data_ref[4]

flops_multi = data_multi[5]
time_multi = data_multi[2]
iterations_multi = data_multi[4]

flops_naive = data_naive[5]
time_naive = data_naive[2]
iterations_naive = data_naive[4]


N_ref = data_ref[3]
flops_ref = data_ref[5]
time_ref = data_ref[2]
iterations_ref = data_ref[4]

flops_seq = data_seq[5]
time_seq = data_seq[2]
iterations_seq = data_seq[4]

flops_optnaive = data_optnaive[5]
time_optnaive = data_optnaive[2]
iterations_optnaive = data_optnaive[4]



plt.figure(1)
#plt.plot(N_ref,flops_ref,'*-')
#plt.plot(N_multi,flops_multi/1000,'*-')
plt.plot(N_naive,flops_naive,'*-')
plt.plot(N_optnaive,flops_optnaive,'*-')
plt.plot(N_seq,flops_seq,'*-')
#plt.plot(N_ref,flops_ref/1000,'*-')
plt.title("")
plt.xlabel("N, grid size")
plt.ylabel("Performance [Mflops/s]")
#plt.ylim(0,50)
plt.legend(["Sequential, GPU", "Naive, GPU"]) 
    
plt.show()


    


