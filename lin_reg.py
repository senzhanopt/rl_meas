from data_process import *
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import numpy as np
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
import time

algo = ["lin", "lasso", "ridge", "pls"][2]

np.random.seed(42)

start, mid, end = 0, 338, 366
load_p = prof[('load', 'p_mw')].iloc[start*96:end*96,:].to_numpy() * 1E3
load_q = prof[('load', 'q_mvar')].iloc[start*96:end*96,:].to_numpy() * 1E3

# match sgen and storage profiles
sgen_p = sb.get_absolute_profiles_from_relative_profiles(net, 'sgen', 'sn_mva').iloc[start*96:end*96,:].to_numpy() * 1E3
storage_p = sb.get_absolute_profiles_from_relative_profiles(net, 'storage', 'sn_mva').iloc[start*96:end*96,:].to_numpy() * 1E3

# power flow
vm = np.ones(((end-start)*96, n_bus-1))
net_load = np.zeros(((end-start)*96, 2*n_bus-2))
for t in range(96*(end-start)):
    p_load = load_p[t,:]
    q_load = load_q[t,:]
    p_sgen = sgen_p[t,:] * np.random.uniform(0.8,1.0,n_pv)
    q_sgen = sgen_p[t,:] * np.random.uniform(-0.2,0.2,n_pv)
    p_storage = storage_p[t,:] * np.random.uniform(0.8,1.0,n_storage)
    q_storage = storage_p[t,:] * np.random.uniform(-0.2,0.2,n_storage)
    dict_return = powerflow(p_load, q_load, p_sgen, q_sgen, p_storage, q_storage)
    vm[t,:] = dict_return["v"]
    net_load[t, :] = np.concatenate((dict_return["p_net"][1:],dict_return["q_net"][1:]))

net_load_train = net_load[:mid*96,:]
vm_train = vm[:mid*96,:]
net_load_test = net_load[mid*96:,:]
vm_test = vm[mid*96:,:]

start_time = time.time()

if algo == "lasso" or algo == "ridge":
    list_mae = []
    list_mae_train = []
    list_coef = []
    list_intercept = []
    for idx, val in enumerate([10.0,5.0,2.0,1.0,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.0]):
        print(f"Regularization parameter is {val} ........................")
        if algo == "lasso":
            ridge = Lasso(alpha=val)
        elif algo == "ridge":
            ridge = Ridge(alpha=val)
        ridge.fit(net_load_train/1e1, (vm_train-1.0)*1e1)
        coef = ridge.coef_ /1e2
        intercept = ridge.intercept_/1e2
        vm_pred = intercept + net_load_test @ coef.T + 1.0
        vm_pred_train = intercept + net_load_train @ coef.T + 1.0
        mae = np.average(np.abs(vm_test-vm_pred))
        mae_train = np.average(np.abs(vm_train-vm_pred_train))
        list_mae.append(mae)
        list_mae_train.append(mae_train)
        list_coef.append(coef)
        list_intercept.append(intercept)


if algo == "pls":  
    list_mae = []
    list_coef = []
    list_mae_train = []
    list_intercept = []
    for idx, val in enumerate([6*i for i in range(1,33)]):
        print(f"No. of components is {val} ........................")
        pls = PLSRegression(n_components=val)
        pls.fit(net_load_train/1e1, (vm_train-1.0)*1e1)
        coef = pls.coef_ /1e2
        intercept = pls.intercept_/1e2
        vm_pred = intercept + net_load_test @ coef.T + 1.0
        vm_pred_train = intercept + net_load_train @ coef.T + 1.0
        mae = np.average(np.abs(vm_test-vm_pred))
        mae_train = np.average(np.abs(vm_train-vm_pred_train))
        list_mae.append(mae)
        list_mae_train.append(mae_train)
        list_coef.append(coef)
        list_intercept.append(intercept)
 

if algo == "lin":
    reg = LinearRegression(fit_intercept=True)
    reg.fit(net_load/1e1, (vm-1.0)*1e1)
    coef = reg.coef_ /1e2
else:   
    idx_best = np.argmin(list_mae)  
    coef = list_coef[idx_best]  
    intercept = list_intercept[idx_best] 
  
print("Time is: \n")
print(time.time() - start_time)

indices_pv = net.sgen.bus.to_list()
indices_pv = [i-1 for i in indices_pv]
mat_R_pv_lr = coef[:,indices_pv]
mat_X_pv_lr = coef[:,[i+n_bus-1 for i in indices_pv]]

indices_storage = net.storage.bus.to_list()
indices_storage = [i-1 for i in indices_storage]
mat_R_storage_lr = coef[:,indices_storage]
mat_X_storage_lr = coef[:,[i+n_bus-1 for i in indices_storage]]

mat_R_pv = pd.read_csv('mat_R_pv.csv', index_col = 0).to_numpy()
mat_X_pv = pd.read_csv('mat_X_pv.csv', index_col = 0).to_numpy()
    

pd.DataFrame(mat_R_storage_lr).to_csv('mat_R_storage_lr.csv')
pd.DataFrame(mat_X_storage_lr).to_csv('mat_X_storage_lr.csv')
pd.DataFrame(mat_R_pv_lr).to_csv('mat_R_pv_lr.csv')
pd.DataFrame(mat_X_pv_lr).to_csv('mat_X_pv_lr.csv')


ax1 = sns.heatmap(mat_R_pv_lr, vmax = 0.0008, vmin = 0.0)
ax1.set_xlabel("PV index")
ax1.set_ylabel("Bus index")
plt.savefig('mat_R_pv_lr.pdf', bbox_inches = 'tight')
plt.show()

ax2 = sns.heatmap(mat_R_pv, vmax = 0.0008, vmin=0.0)
ax2.set_xlabel("PV index")
ax2.set_ylabel("Bus index")
plt.savefig('mat_R_pv.pdf', bbox_inches = 'tight')
plt.show()

ax3 = sns.heatmap(mat_X_pv_lr, vmax = 0.0004,  vmin = 0.0)
ax3.set_xlabel("PV index")
ax3.set_ylabel("Bus index")
plt.savefig('mat_X_pv_lr.pdf', bbox_inches = 'tight')
plt.show()

ax2 = sns.heatmap(mat_X_pv,  vmax = 0.0004, vmin=0.0)
ax2.set_xlabel("PV index")
ax2.set_ylabel("Bus index")
plt.savefig('mat_X_pv.pdf', bbox_inches = 'tight')
plt.show()


if algo != "lin":
    plt.plot(list_mae_train, label = "train")
    plt.plot(list_mae, label = "test")
    plt.legend()
    plt.show()

