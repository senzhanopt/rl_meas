from data_process import *
from sklearn.linear_model import Ridge, Lasso
import numpy as np

# power flow with default storage profiles
vm = np.ones(((end-start-7)*96, n_bus-1))
net_load_p = np.ones(((end-start-7)*96, n_bus-1))
net_load_q = np.ones(((end-start-7)*96, n_bus-1))
for t in range(96*(end-start-7)):
    net.load.p_mw = load_p[t,:] * 1E-3
    net.load.q_mvar = load_q[t,:] * 1E-3
    net.sgen.p_mw = sgen_p[t,:] * 1E-3
    net.sgen.q_mvar = sgen_p[t,:] * 1E-3 * np.random.uniform(-1,1,n_pv)
    net.storage.p_mw = storage_p[t,:] * 1E-3
    net.storage.q_mvar = storage_p[t,:] * 1E-3 * np.random.uniform(-1,1,n_storage)
    pp.runpp(net)
    vm[t,:] = net.res_bus.vm_pu.to_numpy()[1:]
    net_load_p[t, :] = net.res_bus.p_mw.to_numpy()[1:] * 1E3
    net_load_q[t, :] = net.res_bus.q_mvar.to_numpy()[1:] * 1E3
    
net_load = np.concatenate((net_load_p, net_load_q), axis = 1)

reg = Ridge(alpha=100).fit(net_load, vm)
coef = reg.coef_ 
indices_storage = net.storage.bus.to_list()
mat_R_storage_lr = coef[:,indices_storage]
mat_X_storage_lr = coef[:,[i+n_bus-1 for i in indices_storage]]

indices_pv = net.sgen.bus.to_list()
mat_R_pv_lr = coef[:,indices_pv]
mat_X_pv_lr = coef[:,[i+n_bus-1 for i in indices_pv]]

pd.DataFrame(mat_R_storage_lr).to_csv('mat_R_storage_lr.csv')
pd.DataFrame(mat_X_storage_lr).to_csv('mat_X_storage_lr.csv')
pd.DataFrame(mat_R_pv_lr).to_csv('mat_R_pv_lr.csv')
pd.DataFrame(mat_X_pv_lr).to_csv('mat_X_pv_lr.csv')

mat_R_storage = pd.read_csv('mat_R_storage.csv', index_col = 0).to_numpy()
mat_X_storage = pd.read_csv('mat_X_storage.csv', index_col = 0).to_numpy()
mag_R_storage = mat_R_storage_lr/mat_R_storage
mag_X_storage = mat_X_storage_lr/mat_X_storage

mat_R_pv = pd.read_csv('mat_R_pv.csv', index_col = 0).to_numpy()
mat_X_pv = pd.read_csv('mat_X_pv.csv', index_col = 0).to_numpy()
mag_R_pv = mat_R_pv_lr/mat_R_pv
mag_X_pv = mat_X_pv_lr/mat_X_pv

# test
vm_test = np.ones((7*96, n_bus-1))
vm_lr = np.ones((7*96, n_bus-1))
for t in range(7*96):
    net.load.p_mw = load_p[t+(end-start-7)*96,:] * 1E-3
    net.load.q_mvar = load_q[t+(end-start-7)*96,:] * 1E-3
    net.sgen.p_mw = sgen_p[t+(end-start-7)*96,:] * 1E-3
    net.sgen.q_mvar = 0.0
    net.storage.p_mw = storage_p[t+(end-start-7)*96,:] * 1E-3
    net.storage.q_mvar = 0.0
    pp.runpp(net)
    vm_test[t,:] = net.res_bus.vm_pu.to_numpy()[1:]
    vm_lr[t,:] = 1.0 + coef[:,:n_bus-1] @ (net.res_bus.p_mw.to_numpy()[1:] * 1E3) +\
                       coef[:,n_bus-1:] @ (net.res_bus.q_mvar.to_numpy()[1:] * 1E3)

mag_vm = vm_test/vm_lr
