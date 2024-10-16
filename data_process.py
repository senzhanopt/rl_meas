import simbench as sb
import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(['science', 'nature'])

# simulation parameters
start, end = 240,268
n_timesteps = (end-start)*96
n_itr = 15 # one-minute control resolution
n_pv = 48 # 50% nodes have a pv
n_storage = 19 # 20% nodes have a battery
list_bus_visual = [55,66,59,70]

# read pp net object from simbench
net = sb.get_simbench_net('1-LV-rural2--2-no_sw')
net.ext_grid.vm_pu = 1.0
net.trafo.tap_pos = 0
net.sgen = pd.read_excel('sgen.xlsx', index_col = 0).iloc[0:n_pv,:]
net.storage = pd.read_excel('storage.xlsx', index_col = 0).iloc[0:n_storage,:]
pp.plotting.to_html(net, 'grid.html')
n_bus = len(net.bus) # 97 bus

# read load profiles
prof = sb.get_absolute_values(net, 1)
load_p = prof[('load', 'p_mw')].iloc[start*96:end*96,:].to_numpy()
load_q = prof[('load', 'q_mvar')].iloc[start*96:end*96,:].to_numpy()

# match sgen and storage profiles
sgen_p = sb.get_absolute_profiles_from_relative_profiles(net, 'sgen', 'sn_mva').iloc[start*96:end*96,:].to_numpy()
storage_p = sb.get_absolute_profiles_from_relative_profiles(net, 'storage', 'sn_mva').iloc[start*96:end*96,:].to_numpy()

# power flow with default storage profiles
vm = np.ones(((end-start)*96, n_bus))
loading_trafo = np.zeros((end-start)*96)
for t in range(96*(end-start)):
    net.load.p_mw = load_p[t,:]
    net.load.q_mvar = load_q[t,:]
    net.sgen.p_mw = sgen_p[t,:]
    net.sgen.q_mvar = 0
    net.storage.p_mw = storage_p[t,:]
    net.storage.q_mvar = 0
    pp.runpp(net)
    vm[t,:] = net.res_bus.vm_pu.to_numpy()
    loading_trafo[t] = net.res_trafo.loading_percent[0]
    
    
for b in list_bus_visual:
    plt.plot(vm[:,b], label = f'bus {b}')
plt.legend()
plt.show()
    
plt.plot(loading_trafo, label = "MV/LV transformer")
plt.legend()
plt.show()




