#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:54:02 2022

@author: palinakroyer
"""


import numpy as np
import matplotlib.pyplot as plt


data_ref = np.loadtxt("Data/Jacobi_ref/jacobi_ref.dat",unpack=True)
data_seq = np.loadtxt("Data/Jacobi_seq/jacobi_seq.dat",unpack=True)


N_ref = data_ref[3]
N_seq = data_seq[3]


flops_ref = data_ref[5]
time_ref = data_ref[2]
iterations_ref = data_ref[4]

flops_seq = data_seq[5]
time_seq = data_seq[2]
iterations_seq = data_seq[4]





plt.figure(1)
plt.plot(N_ref,flops_ref,'*-')
plt.plot(N_seq,flops_seq,'*-')
plt.title("")
plt.xlabel("N")
plt.ylabel("Performance [Mflops/s]")
plt.legend(["Reference", "Sequential, GPU"]) 
    
plt.show()


    


