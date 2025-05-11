from data_process import *


#%% network sensitivity
mat_R_storage = np.zeros((n_bus-1, n_storage))
mat_X_storage = np.zeros((n_bus-1, n_storage))
mat_R_pv = np.zeros((n_bus-1, n_pv))
mat_X_pv = np.zeros((n_bus-1, n_pv))

# base case
net.load.p_mw = 0.0
net.load.q_mvar = 0.0
net.sgen.p_mw = S_pv * 0.9E-3
net.sgen.q_mvar = 0.0
net.storage.p_mw = 0.0
net.storage.q_mvar = 0.0
pp.runpp(net)
v_base = net.res_bus.vm_pu.to_numpy()[1:]


for b in range(n_storage):
    net.storage.loc[b, 'p_mw'] = 1E-3
    pp.runpp(net)
    v_current = net.res_bus.vm_pu.to_numpy()[1:]
    mat_R_storage[:,b] = (v_current - v_base)
    net.storage.loc[b, 'p_mw'] = 0.0 # recover the base case

for b in range(n_storage):
    net.storage.loc[b, 'q_mvar'] = 1E-3
    pp.runpp(net)
    v_current = net.res_bus.vm_pu.to_numpy()[1:]
    mat_X_storage[:,b] = (v_current - v_base)
    net.storage.loc[b, 'q_mvar'] = 0.0 # recover the base case

for b in range(n_pv):
    net.sgen.loc[b, 'p_mw'] += 1E-3
    pp.runpp(net)
    v_current = net.res_bus.vm_pu.to_numpy()[1:]
    mat_R_pv[:,b] = (v_current - v_base)
    net.sgen.loc[b, 'p_mw'] -= 1E-3 # recover the base case

for b in range(n_pv):
    net.sgen.loc[b, 'q_mvar'] += 1E-3
    pp.runpp(net)
    v_current = net.res_bus.vm_pu.to_numpy()[1:]
    mat_X_pv[:,b] = (v_current - v_base)
    net.sgen.loc[b, 'q_mvar'] -= 1E-3 # recover the base case


pd.DataFrame(mat_R_storage).to_csv('mat_R_storage.csv')
pd.DataFrame(mat_X_storage).to_csv('mat_X_storage.csv')
pd.DataFrame(mat_R_pv).to_csv('mat_R_pv.csv')
pd.DataFrame(mat_X_pv).to_csv('mat_X_pv.csv')

