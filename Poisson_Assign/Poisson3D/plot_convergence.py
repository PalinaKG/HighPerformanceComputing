# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:35:33 2022

@author: Bernhard
"""

import numpy as np
import matplotlib.pyplot as plt


data_5 = np.loadtxt("Convergence_Data/convergence_5.dat",unpack=True)
data_50 = np.loadtxt("Convergence_Data/convergence_50.dat",unpack=True)
data_120 = np.loadtxt("Convergence_Data/convergence_120.dat",unpack=True)

data_5_gs = np.loadtxt("Convergence_Data/convergence_gs_5.dat",unpack=True)
data_50_gs = np.loadtxt("Convergence_Data/convergence_gs_50.dat",unpack=True)
data_120_gs = np.loadtxt("Convergence_Data/convergence_gs_120.dat",unpack=True)





plt.figure(1)
plt.plot(data_120[0],data_120[1],'b-')
plt.plot(data_120_gs[0],data_120_gs[1],'r-')




plt.title("Convergience for Jacobi and Gauss-Seidel implementations")
plt.xlabel("Iterations")
plt.ylabel("Convergience")
plt.legend(["Jacobi", "Gauss-Seidel"]) 
plt.axis([0, 5000, 0, 10])







    


