from ofo import *
import matplotlib.pyplot as plt

list_apc = []
list_e = []

for i in range(7):
    e = 0.2 + 0.1 * i
    pv_max, pv_gen, avv, alv = main(e  = e, w2 = 0.02, epsilon = 1.0)
    list_apc.append(pv_max-pv_gen)
    list_e.append(e)
    


list_apc2 = []
list_eps = [0.01,0.1,0.2,0.5,0.8,1.0,2.0]
for epsilon in list_eps:
    pv_max, pv_gen, avv, alv = main(e  = 0.2, w2 = 0.0, epsilon = epsilon)
    list_apc2.append(pv_max-pv_gen)

plt.figure(figsize=(2,3))  
plt.plot(list_e, list_apc, marker = '.')
plt.ylabel('PV energy curtailment (kWh)')
plt.xlabel('$e_i/\overline{E}_i$')
plt.savefig('e.pdf',bbox_inches='tight')
plt.show()
  
plt.figure(figsize=(2,3))   
plt.plot(list_eps, list_apc2, marker = '.')
plt.ylabel('PV energy curtailment (kWh)')
plt.xlabel('$\gamma$')
plt.savefig('gamma.pdf',bbox_inches='tight')
plt.show()