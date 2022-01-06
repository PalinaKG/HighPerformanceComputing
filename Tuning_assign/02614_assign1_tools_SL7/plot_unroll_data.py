# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:48:39 2022

@author: Bernhard
"""

import numpy as np
import matplotlib.pyplot as plt

perms = ["mnk", "nmk", "nkm", "mkn", "kmn", "knm"]
perms = ["mnk"]
unrolls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
legend = []

for perm in perms:
    plt.figure()
    plt.title(perm)
    for n in unrolls:
        print(perm+"_"+str(n)+".dat")
        data = np.loadtxt("Unroll_Data\\"+perm+"_"+str(n)+".dat",unpack=True)
        memory = data[0]
        legend.append("Unroll= " + str(n))
        
        mf = data[1]

        plt.plot(memory,mf,'*-')

    plt.xlabel("Memory [kbytes]")
    plt.ylabel("Performance [Mflops/s]")
    plt.legend(legend)
