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

plt.figure(1)
plt.plot(threads,speedup,'*-')
plt.plot([0, 24], [0, 24],'-')
plt.title("Simple OpenMP for Gauss-Seidel")
plt.xlabel("Threads")
plt.ylabel("Speed up")
