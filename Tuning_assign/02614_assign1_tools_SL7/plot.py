import matplotlib.pyplot as plt 
import numpy as np

mkn_data = np.loadtxt('Control_Data/mkn.dat')
nkm_data = np.loadtxt('Control_Data/nkm.dat')
knm_data = np.loadtxt('Control_Data/knm.dat')
mnk_data = np.loadtxt('Control_Data/mnk.dat')
nmk_data = np.loadtxt('Control_Data/nmk.dat')
kmn_data = np.loadtxt('Control_Data/kmn.dat')
#lib_data = np.loadtxt('Control_Data/lib.dat')

plt.plot(mkn_data[:,0], mkn_data[:,1], 'ro-', label='MKN') 
plt.plot(nkm_data[:,0], nkm_data[:,1], 'go-', label='NKM') 
plt.plot(knm_data[:,0], knm_data[:,1], 'bo-', label='KNM') 
plt.plot(mnk_data[:,0], mnk_data[:,1], 'co-', label='MNK') 
plt.plot(nmk_data[:,0], nmk_data[:,1], 'mo-', label='NMK') 
plt.plot(kmn_data[:,0], kmn_data[:,1], 'yo-', label='KMN') 
#plt.plot(lib_data[:,0], lib_data[:,1], 'ko-', label='lib') 

plt.xscale('log', base=2)
plt.axvline(x=32)
plt.axvline(x=256)
plt.axvline(x=30720)
plt.legend(loc='upper left')


plt.show()