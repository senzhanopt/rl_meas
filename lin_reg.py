from data_process import *
from sklearn.linear_model import Ridge, Lasso

# power flow with default storage profiles
vm = np.ones(((end-start-7)*96, n_bus-1))
net_load_p = np.ones(((end-start-7)*96, n_bus-1))
net_load_q = np.ones(((end-start-7)*96, n_bus-1))
for t in range(96*(end-start-7)):
    net.load.p_mw = load_p[t,:] * 1E-3
    net.load.q_mvar = load_q[t,:] * 1E-3
    net.sgen.p_mw = sgen_p[t,:] * 1E-3
    net.sgen.q_mvar = 0.0
    net.storage.p_mw = storage_p[t,:] * 1E-3
    net.storage.q_mvar = 0.0
    pp.runpp(net)
    vm[t,:] = net.res_bus.vm_pu.to_numpy()[1:]
    net_load_p[t, :] = net.res_bus.p_mw.to_numpy()[1:] * 1E3
    net_load_q[t, :] = net.res_bus.q_mvar.to_numpy()[1:] * 1E3
    
net_load = np.concatenate((net_load_p, net_load_q), axis = 1)

reg = Ridge(alpha=1.0).fit(net_load/1E1, (vm-1.0)*1E2)
coef = reg.coef_ / 1E3
indices_storage = net.storage.bus.to_list()
mat_R_storage_lr = coef[:,indices_storage]
mat_X_storage_lr = coef[:,[i+n_bus-1 for i in indices_storage]]

pd.DataFrame(mat_R_storage_lr).to_csv('mat_R_storage_lr.csv')
pd.DataFrame(mat_X_storage_lr).to_csv('mat_X_storage_lr.csv')

mat_R_storage = pd.read_csv('mat_R_storage.csv', index_col = 0).to_numpy()
mat_X_storage = pd.read_csv('mat_X_storage.csv', index_col = 0).to_numpy()
mag_R = mat_R_storage_lr/mat_R_storage
mag_X = mat_X_storage_lr/mat_X_storage

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