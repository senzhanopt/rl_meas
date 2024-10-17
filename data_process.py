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
soc_min, soc_max, soc_init = 0.2, 0.8, 0.5

# read pp net object from simbench
net = sb.get_simbench_net('1-LV-rural2--2-no_sw')
net.ext_grid.vm_pu = 1.0
net.trafo.tap_pos = 0
net.sgen = pd.read_excel('sgen.xlsx', index_col = 0).iloc[0:n_pv,:]
net.storage = pd.read_excel('storage.xlsx', index_col = 0).iloc[0:n_storage,:]
#pp.plotting.to_html(net, 'grid.html')
n_bus = len(net.bus) # 97 bus

# read load profiles
prof = sb.get_absolute_values(net, 1)
load_p = prof[('load', 'p_mw')].iloc[start*96:end*96,:].to_numpy()
load_q = prof[('load', 'q_mvar')].iloc[start*96:end*96,:].to_numpy()

# match sgen and storage profiles
sgen_p = sb.get_absolute_profiles_from_relative_profiles(net, 'sgen', 'sn_mva').iloc[start*96:end*96,:].to_numpy()
storage_p = sb.get_absolute_profiles_from_relative_profiles(net, 'storage', 'sn_mva').iloc[start*96:end*96,:].to_numpy()

# power flow with default storage profiles
vm = np.ones(((end-start)*96, n_bus-1))
loading_trafo = np.zeros((end-start)*96)
for t in range(96*(end-start)):
    net.load.p_mw = load_p[t,:]
    net.load.q_mvar = load_q[t,:]
    net.sgen.p_mw = sgen_p[t,:]
    net.sgen.q_mvar = 0.0
    net.storage.p_mw = storage_p[t,:]
    net.storage.q_mvar = 0.0
    pp.runpp(net)
    vm[t,:] = net.res_bus.vm_pu.to_numpy()[1:]
    loading_trafo[t] = net.res_trafo.loading_percent[0]
    
for b in list_bus_visual:
    plt.plot(vm[:,b-1], label = f'bus {b}')
plt.legend()
plt.show()
    
plt.plot(loading_trafo, label = "MV/LV transformer")
plt.legend()
plt.show()


#%% network sensitivity
mat_R_storage = np.zeros((n_bus-1, n_storage))
mat_X_storage = np.zeros((n_bus-1, n_storage))
mat_R_pv = np.zeros((n_bus-1, n_pv))
mat_X_pv = np.zeros((n_bus-1, n_pv))

# base case
net.load.p_mw = 0.0
net.load.q_mvar = 0.0
net.sgen.p_mw = 0.0
net.sgen.q_mvar = 0.0
net.storage.p_mw = 0.0
net.storage.q_mvar = 0.0
pp.runpp(net)
v_base = net.res_bus.vm_pu.to_numpy()[1:]

for b in range(n_storage):
    net.storage.loc[b, 'p_mw'] = 1E-3
    pp.runpp(net)
    v_current = net.res_bus.vm_pu.to_numpy()[1:]
    mat_R_storage[:,b] = v_current - v_base
    net.storage.loc[b, 'p_mw'] = 0.0 # recover the base case

for b in range(n_storage):
    net.storage.loc[b, 'q_mvar'] = 1E-3
    pp.runpp(net)
    v_current = net.res_bus.vm_pu.to_numpy()[1:]
    mat_X_storage[:,b] = v_current - v_base
    net.storage.loc[b, 'q_mvar'] = 0.0 # recover the base case

for b in range(n_pv):
    net.sgen.loc[b, 'p_mw'] = 1E-3
    pp.runpp(net)
    v_current = net.res_bus.vm_pu.to_numpy()[1:]
    mat_R_pv[:,b] = v_current - v_base
    net.sgen.loc[b, 'p_mw'] = 0.0 # recover the base case

for b in range(n_pv):
    net.sgen.loc[b, 'q_mvar'] = 1E-3
    pp.runpp(net)
    v_current = net.res_bus.vm_pu.to_numpy()[1:]
    mat_X_pv[:,b] = v_current - v_base
    net.sgen.loc[b, 'q_mvar'] = 0.0 # recover the base case


