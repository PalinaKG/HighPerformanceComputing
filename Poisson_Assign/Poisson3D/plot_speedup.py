#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:54:02 2022

@author: palinakroyer
"""


import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("Simple_Omp/Data/Simp_omp_gs.dat",unpack=True)


N = data[3]


flops = data[5]
time = data[2]
iterations = data[4]






plt.figure(1)
plt.plot(N,iterations,'*-')
plt.title("")
plt.xlabel("N")
plt.ylabel("Iterations")
