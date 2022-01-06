import matplotlib.pyplot as plt 

plt.plotfile('mkn.dat', delimiter=' ', cols=(0, 1),
             names=('col1', 'col2'), marker='o')

plt.plotfile('nkm.dat', delimiter=' ', cols=(0, 1),
             names=('col1', 'col2'), marker='o', newfig=False)

plt.plotfile('knm.dat', delimiter=' ', cols=(0, 1),
             names=('col1', 'col2'), marker='o', newfig=False)

plt.plotfile('mnk.dat', delimiter=' ', cols=(0, 1),
             names=('col1', 'col2'), marker='o', newfig=False)

plt.plotfile('nmk.dat', delimiter=' ', cols=(0, 1),
             names=('col1', 'col2'), marker='o', newfig=False)

plt.plotfile('kmn.dat', delimiter=' ', cols=(0, 1),
             names=('col1', 'col2'), marker='o', newfig=False)

             

plt.xscale('log', base=2)
plt.axvline(x=32)
plt.axvline(x=256)
plt.axvline(x=30720)


plt.show()