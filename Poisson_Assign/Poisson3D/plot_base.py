#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:54:02 2022

@author: palinakroyer
"""


import numpy as np
import matplotlib.pyplot as plt


data_j = np.loadtxt("Base/base_j.dat",unpack=True)
data_gs = np.loadtxt("Base/base_gs.dat",unpack=True)


N_j = data_j[3]
N_gs = data_gs[3]


flops_j = data_j[5]
time_j = data_j[2]
iterations_j = data_j[4]

flops_gs = data_gs[5]
time_gs = data_gs[2]
iterations_gs = data_gs[4]





plt.figure(1)
plt.plot(N_j,iterations_j,'*-')
plt.plot(N_gs,iterations_gs,'*-')
plt.title("")
plt.xlabel("N")
plt.ylabel("Iterations")
plt.legend(["Jacobi", "Gauss-Seidel"]) 


plt.figure(2)
plt.plot(N_j,time_j,'*-')
plt.plot(N_gs,time_gs,'*-')
plt.xlabel("N")
plt.ylabel("Time [s]")
plt.legend(["Jacobi", "Gauss-Seidel"]) 


plt.figure(3)
plt.plot(N_j,flops_j,'*-')
plt.plot(N_gs,flops_gs,'*-')
plt.xlabel("N")
plt.ylabel("Performance [Mflops/s]")
plt.legend(["Jacobi", "Gauss-Seidel"]) 

    



    


