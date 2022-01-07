import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.pyplot import figure
from matplotlib.ticker import ScalarFormatter

mkn_data = np.loadtxt('Control_Data/mkn.dat')
nkm_data = np.loadtxt('Control_Data/nkm.dat')
knm_data = np.loadtxt('Control_Data/knm.dat')
mnk_data = np.loadtxt('Control_Data/mnk.dat')
nmk_data = np.loadtxt('Control_Data/nmk.dat')
kmn_data = np.loadtxt('Control_Data/kmn.dat')
lib_data = np.loadtxt('Control_Data/lib.dat')

plt.figure()
plt.plot(mkn_data[:,0], mkn_data[:,1], 'ro-', label='mkn') 
plt.plot(nkm_data[:,0], nkm_data[:,1], 'go-', label='nkm') 
plt.plot(knm_data[:,0], knm_data[:,1], 'bo-', label='knm') 
plt.plot(mnk_data[:,0], mnk_data[:,1], 'co-', label='mnk') 
plt.plot(nmk_data[:,0], nmk_data[:,1], 'mo-', label='nmk') 
plt.plot(kmn_data[:,0], kmn_data[:,1], 'yo-', label='kmn')
#plt.plot(lib_data[:,0], lib_data[:,1], 'ko-', label='lib') 

plt.xscale('log', basex=2)
plt.xticks([1, 4, 16, 64, 256, 1024, 4096, 16384, 65536])
plt.axvline(x=32)
plt.axvline(x=256)
plt.axvline(x=30720)
plt.legend(loc='upper left')
plt.title('Permutation comparison without compiler optimization')
plt.xlabel("Memory [kbytes]")
plt.ylabel("Performance [Mflops/s]")


plt.figure()
plt.plot(lib_data[:,0], lib_data[:,1], 'ko-', label='lib') 
plt.plot(mnk_data[:,0], mnk_data[:,1], 'co-', label='nat') 

plt.xscale('log', basex=2)
plt.xticks([1, 4, 16, 64, 256, 1024, 4096, 16384, 65536])
plt.axvline(x=32)
plt.axvline(x=256)
plt.axvline(x=30720)
plt.legend(loc='upper left')
plt.title('Permutation comparison without compiler optimization')
plt.xlabel("Memory [kbytes]")
plt.ylabel("Performance [Mflops/s]")

plt.show()

