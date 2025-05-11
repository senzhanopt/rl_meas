from ofo import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

list_apc = []
list_avv = []
list_alv = []

list_eps = [0.05,0.1,0.2,0.5,0.8,1.0,2.0]
for i in range(7):
    e = 0.2 + 0.1 * i
    for j in range(7):     
        pv_max, pv_gen, avv, alv = main(e  = e, w2 = 0.02, epsilon = list_eps[j])
        list_apc.append(pv_max-pv_gen)
        list_avv.append(avv)
        list_alv.append(alv)
    


X, Y = np.meshgrid(list_eps, [0.2+i*0.1 for i in range(7)])

# Reshape the list_apc values into a 2D grid for plotting
Z = np.array(list_apc).reshape(7,7)

# Plot
fig = plt.figure(figsize=(7.5,4.5))
ax = fig.add_subplot(111, projection='3d')

# Create a surface plot
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha = 0.9)

# Add labels and a color bar
ax.set_xlabel('$\gamma$')
ax.set_ylabel('$e_i/\overline{E}_i$')
ax.set_zlabel('PV energy curtailment (kWh)')
fig.colorbar(surf)
plt.savefig('lyap.pdf',bbox_inches='tight')
plt.show()