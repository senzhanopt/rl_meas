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
itr_length = 0.25 / n_itr # in hour
n_pv = 48 # 50% nodes have a pv
n_storage = 19 # 20% nodes have a battery
list_bus_visual = [55,66,59,70]
soc_min, soc_max, soc_init = 0.2, 0.8, 0.5
eta_ch, eta_dis = 0.95, 0.95

# read pp net object from simbench
net = sb.get_simbench_net('1-LV-rural2--2-no_sw')
net.ext_grid.vm_pu = 1.0
net.trafo.tap_pos = 0
net.sgen = pd.read_excel('sgen.xlsx', index_col = 0).iloc[0:n_pv,:]
net.storage = pd.read_excel('storage.xlsx', index_col = 0).iloc[0:n_storage,:]
#pp.plotting.to_html(net, 'grid.html')
n_bus = len(net.bus) # 97 bus
S_pv = net.sgen.sn_mva.to_numpy() * 1E3
S_storage = net.storage.sn_mva.to_numpy() * 1E3
E_storage = net.storage.max_e_mwh.to_numpy() * 1E3

# read load profiles
prof = sb.get_absolute_values(net, 1)
load_p = prof[('load', 'p_mw')].iloc[start*96:end*96,:].to_numpy() * 1E3
load_q = prof[('load', 'q_mvar')].iloc[start*96:end*96,:].to_numpy() * 1E3

# match sgen and storage profiles
sgen_p = sb.get_absolute_profiles_from_relative_profiles(net, 'sgen', 'sn_mva').iloc[start*96:end*96,:].to_numpy() * 1E3
storage_p = sb.get_absolute_profiles_from_relative_profiles(net, 'storage', 'sn_mva').iloc[start*96:end*96,:].to_numpy() * 1E3



