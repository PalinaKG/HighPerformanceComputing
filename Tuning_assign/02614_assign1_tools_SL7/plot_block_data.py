# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:34:27 2022

@author: Bernhard
"""

import numpy as np
import matplotlib.pyplot as plt

sizes = ["12", "24", "36", "48", "75", "102", "124", "200", "250", 
         "350", "480", "730", "800", "1000", "1200", "1500"]

block_sizes = [2,4,8,16,32,64,128,256,512,1028,2056]

for size in sizes:
    data = np.loadtxt("blk_"+size+".dat",unpack=True)

    mf = data[1]


    plt.figure()
    plt.plot(block_sizes,mf,'*-')


    plt.title("Matrix size: " + size)
    plt.xlabel("Block size")
    plt.ylabel("Performance [Mflops/s]")