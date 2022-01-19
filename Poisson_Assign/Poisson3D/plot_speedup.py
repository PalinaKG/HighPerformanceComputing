#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:54:02 2022

@author: palinakroyer
"""


import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("Simple_Omp/Data/Simp_omp_gs.dat",unpack=True)

threads = [1, 2, 3, 4, 5, 7, 10, 12, 15, 17, 20, 24]
time = data[2]

speedup = time[0]/time
print(speedup)

plt.figure(1)
plt.plot(threads,speedup,'*-')
plt.plot([0, 24], [0, 24],'-')
plt.title("Simple OpenMP for Gauss-Seidel")
plt.xlabel("Threads")
plt.ylabel("Speed up")
plt.legend(["Ideal", "OpenMp doacross"])

plt.figure(2)
data = np.loadtxt("Simple_Omp/Data/Simp_omp_j.dat",unpack=True)
data1 = np.loadtxt("Rewrite/Data/Simp_omp_jV1.dat",unpack=True)
data2 = np.loadtxt("RewriteMP/Data/Simp_omp_jV2.dat",unpack=True)

threads = [1, 2, 3, 4, 5, 7, 10, 12, 15, 17, 20, 24]
time = data[2]
speedup3 = time[0]/time

time1 = data1[2]
speedup1 = time1[0]/time1

time2 = data2[2]
speedup2 = time2[0]/time2

plt.plot([0, 24], [0, 24],'-')
plt.plot(threads,speedup3,'*-')
plt.plot(threads,speedup1,'*-')
plt.plot(threads,speedup2,'*-')
plt.title("OpenMP for Jacobi")
plt.xlabel("Threads")
plt.ylabel("Speed up")
plt.legend(["Ideal", "OpenMP 1", "OpenMP 2", "OpenMp 3"])


## NONE OPTIMZED#####


data = np.loadtxt("Simple_Omp/Data/Simp_omp_gs_op.dat",unpack=True)

threads = [1, 2, 3, 4, 5, 7, 10, 12, 15, 17, 20, 24]
time = data[2]

speedup = time[0]/time

plt.figure(3)
plt.plot(threads,speedup,'*-')
plt.plot([0, 24], [0, 24],'-')
plt.title("None opt, OpenMP for Gauss")
plt.xlabel("Threads")
plt.ylabel("Speed up")
plt.legend(["Ideal", "OpenMp doacross"])

plt.figure(4)
data = np.loadtxt("Simple_Omp/Data/Simp_omp_j_op.dat",unpack=True)
data1 = np.loadtxt("Rewrite/Data/Simp_omp_jV1_op.dat",unpack=True)
data2 = np.loadtxt("RewriteMP/Data/Simp_omp_jV2_op.dat",unpack=True)

threads = [1, 2, 3, 4, 5, 7, 10, 12, 15, 17, 20, 24]
time = data[2]
speedup = time[0]/time

time1 = data1[2]
speedup1 = time1[0]/time1

time2 = data2[2]
speedup2 = time2[0]/time2

plt.plot([0, 24], [0, 24],'-')
plt.plot(threads,speedup,'*-')
plt.plot(threads,speedup1,'*-')
plt.plot(threads,speedup2,'*-')
plt.title("None opt, OpenMP for Jacobi")
plt.xlabel("Threads")
plt.ylabel("Speed up")
plt.legend(["Ideal", "OpenMP 1", "OpenMP 2", "OpenMp 3"])
