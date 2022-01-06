# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:35:33 2022

@author: Bernhard
"""

import numpy as np
import matplotlib.pyplot as plt

perms = ["mnk", "nmk", "nkm", "mkn", "kmn", "knm"]

for perm in perms:
    data_O2 = np.loadtxt(perm+"_-O2.dat",unpack=True)
    data_O3 = np.loadtxt(perm+"_-O3.dat",unpack=True)
    data_Ofast = np.loadtxt(perm+"_-Ofast.dat",unpack=True)

    memory = data_O2[0]
    memory_O3 = data_O3[0]
    memory_Ofast = data_Ofast[0]
    
    mf_O2 = data_O2[1]
    mf_O3 = data_O3[1]
    mf_Ofast = data_Ofast[1]

    plt.figure()
    plt.plot(memory,mf_O2,'*-')
    plt.plot(memory_O3,mf_O3,'*-')
    plt.plot(memory_Ofast,mf_Ofast,'*-')

    plt.title(perm)
    plt.xlabel("Memory [kbytes]")
    plt.ylabel("Performance [Mflops/s]")
    plt.legend(["-O2","-O3","-Ofast"])
    


