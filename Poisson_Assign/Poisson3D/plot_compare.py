# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:34:26 2022

@author: Bernhard
"""


import numpy as np
import matplotlib.pyplot as plt



### gaus ###
data1 = np.loadtxt("Simple_Omp/Data/Simp_omp_gs_perf_12.dat",unpack=True)
data2 = np.loadtxt("Base/base_gs.dat",unpack=True)

itera1 = data1[4]
flops1 = data1[5]
grid1 = data1[3]
print(grid1)

itera2 = data2[4]
flops2 = data2[5]
grid2 = data2[3]

plt.figure(1)
plt.plot(grid1,itera1,'*-')
plt.plot(grid2,itera1,'*-')
plt.title("Convergence comparison for diffrent Gauss")
plt.xlabel("Grid size, N")
plt.ylabel("Iterations")
plt.legend(["Simple Gauss OpenMp" , "Sequential Gauss"])

plt.figure(2)
plt.plot(grid1,flops1,'*-')
plt.plot(grid2,flops2,'*-')

plt.title("Performace comparison for diffrent Gauss")
plt.xlabel("Grid size, N")
plt.ylabel("Performance [Mflops/s]")
plt.legend(["Simple Gauss OpenMp" , "Sequential Gauss"])


### jacobi ###
data_j = np.loadtxt("Base/base_j.dat",unpack=True)
data1 = np.loadtxt("Simple_Omp/Data/Simp_omp_j_perf_12.dat",unpack=True)
data2 = np.loadtxt("Rewrite/Data/Simp_omp_jV1_perf_12.dat",unpack=True)
data3 = np.loadtxt("RewriteMP/Data/Simp_omp_jV2_perf_12.dat",unpack=True)

iteraj = data_j[4]
flopsj = data_j[5]
gridj = data_j[3]

itera1 = data1[4]
flops1 = data1[5]
grid1 = data1[3]

itera2 = data2[4]
flops2 = data2[5]
grid2 = data2[3]

itera3 = data3[4]
flops3 = data3[5]
grid3 = data3[3]

plt.figure(3)
plt.plot(gridj,iteraj,'*-')
plt.plot(grid1,itera1,'*-')
plt.plot(grid2,itera1,'*-')
plt.plot(grid3,itera3,'*-')
plt.title("Convergence comparison for diffrent Jacobi")
plt.xlabel("Grid size, N")
plt.ylabel("Iterations")
plt.legend(["Sequential Jacobi" , "OpenMP 1", "OpenMP 2", "OpenMp 3"])

plt.figure(4)
plt.plot(gridj,flopsj,'*-')
plt.plot(grid1,flops1,'*-')
plt.plot(grid2,flops2,'*-')
plt.plot(grid3,flops3,'*-')

plt.title("Performace comparison for diffrent Jacobi")
plt.xlabel("Grid size, N")
plt.ylabel("Performance [Mflops/s]")
plt.legend(["Sequential Jacobi" , "OpenMP 1", "OpenMP 2", "OpenMp 3"])