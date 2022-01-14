# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:35:33 2022

@author: Bernhard
"""

import numpy as np
import matplotlib.pyplot as plt


data_O2 = np.loadtxt("Compile_Data/jacobi_base-O2.dat",unpack=True)
data_O3 = np.loadtxt("Compile_Data/jacobi_base-O3.dat",unpack=True)
data_Ofast = np.loadtxt("Compile_Data/jacobi_base-Ofast.dat",unpack=True)

data_floop = np.loadtxt("Compile_Data/jacobi_base-Ofast-floop-interchange.dat",unpack=True)
data_fpeel = np.loadtxt("Compile_Data/jacobi_base-Ofast-fpeel-loops.dat",unpack=True)
data_funroll = np.loadtxt("Compile_Data/jacobi_base-Ofast-funroll-loops.dat",unpack=True)
data_funsafe = np.loadtxt("Compile_Data/jacobi_base-Ofast-funsafe-loop-optimizations.dat",unpack=True)

memory = data_O2[1]

mf_O2 = data_O2[5]
mf_O3 = data_O3[5]
mf_Ofast = data_Ofast[5]

mf_floop = data_floop[5]
mf_fpeel = data_fpeel[5]
mf_funroll = data_funroll[5]
mf_funsafe = data_funsafe[5]


memory = data_O2[1]
N = data_O2[3]


flops = data_Ofast[0]
time = data_Ofast[2]
iterations = data_Ofast[4]




plt.figure(1)

plt.plot(memory,mf_O2,'*-')
plt.plot(memory,mf_O3,'*-')
plt.plot(memory,mf_Ofast,'*-')



plt.title("Performance for different compiler options")
plt.xlabel("Memory [kbytes]")
plt.ylabel("Performance [Mflops/s]")
plt.legend(["O2", "O3", "Ofast"]) 
plt.xscale('log', base=2)
plt.axvline(x=32)
plt.axvline(x=256)
plt.axvline(x=30720)  



plt.figure(2)
plt.plot(memory,mf_Ofast,'*-')
plt.plot(memory,mf_floop,'*-')
plt.plot(memory,mf_fpeel,'*-')
plt.plot(memory,mf_funroll,'*-')
plt.plot(memory,mf_funsafe,'*-')



plt.title("Performance for different compiler options")
plt.xlabel("Memory [kbytes]")
plt.ylabel("Performance [Mflops/s]")
plt.legend(["O2", "O3", "Ofast"])
plt.legend(["Ofast","floop-interchange","fpeel-loops","funroll-loops","funsafe-loop-optimizations"])
plt.axvline(x=32)
plt.axvline(x=256)
plt.axvline(x=30720)  
plt.xscale('log', base=2)


plt.figure(3)
plt.plot(N,iterations,'*-')

plt.xlabel("Memory [kbytes]")
plt.ylabel("Performance [Mflops]")

plt.figure(4)
plt.plot(N,time,'*-')

plt.xlabel("Memory [kbytes]")
plt.ylabel("Time [s]")

plt.figure(5)
plt.plot(N,flops,'*-')

plt.xlabel("Memory [kbytes]")
plt.ylabel("Flops [Mflops]")

    



    


