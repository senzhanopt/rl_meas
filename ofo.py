from data_process import *
import numpy as np
from tqdm import tqdm
import copy
import time

start_ofo, end_ofo = 21*96, 28*96
load_p = load_p[start_ofo:end_ofo,:]
load_q = load_q[start_ofo:end_ofo,:]
sgen_p = sgen_p[start_ofo:end_ofo,:]
    
def main(epsilon=0.5, e=0.2, w2 = 0.0):
    ## parameters
    beta = 1e5
    rho = 1e2
    alpha = 0.5
    xi = 0.1
    epsilon = epsilon/60
    
    pv_control = True
    lr = True
    pf = 'pgm'
    n_timesteps = end_ofo-start_ofo
    
    if lr:
        mat_R_storage = pd.read_csv('mat_R_storage_lr.csv', index_col = 0).to_numpy()
        mat_X_storage = pd.read_csv('mat_X_storage_lr.csv', index_col = 0).to_numpy()
        mat_R_pv = pd.read_csv('mat_R_pv_lr.csv', index_col = 0).to_numpy()
        mat_X_pv = pd.read_csv('mat_X_pv_lr.csv', index_col = 0).to_numpy()
    else:
        mat_R_storage = pd.read_csv('mat_R_storage.csv', index_col = 0).to_numpy()
        mat_X_storage = pd.read_csv('mat_X_storage.csv', index_col = 0).to_numpy()
        mat_R_pv = pd.read_csv('mat_R_pv.csv', index_col = 0).to_numpy()
        mat_X_pv = pd.read_csv('mat_X_pv.csv', index_col = 0).to_numpy()
    
    
    def projection_pv(P, S, p, q):
    
        '''
        P: active power limit
        S: inverter capacity
        p, q: point to be projected
        Optimal Power Flow Pursuit, appendix B setpoint update
        '''
    
        Q = np.sqrt(S**2-P**2)
        if P == 0.0:
            if q >= S:
                return np.array([0, S])
            elif q <= -S:
                return np.array([0, -S])
            else:
                return np.array([0, q])
        else:
            # region Y
            if (p**2 + q**2 <= S**2) and (p >= 0) and (p <= P):
                return np.array([p, q])
            # region A
            elif (p**2 + q**2 >= S**2) and (p >= 0) and (
                    (q >= Q/P * p) or (q <= -Q/P * p)
                    ):
                return np.array([p, q]) * S / np.sqrt(p**2 + q**2)
            # region B
            elif (q >= Q) and (q <= Q/P * p):
                return np.array([P, Q]) 
            elif (q <= -Q) and (q >= -Q/P * p):
                return np.array([P, -Q]) 
            # region C
            elif (p >= P) and (q <= Q) and (q >= -Q):
                return np.array([P, q])
            # region D
            elif (p <= 0) and (q >= -S) and (q <= S):
                return np.array([0, q])
            # region E
            elif (p <= 0) and (q >= S):
                return np.array([0, S])
            elif (p <= 0) and (q <= -S):
                return np.array([0, -S])
            else:
                print(f"P, S, p, q values are {P}, {S}, {p}, {q}.")
                raise Exception("Projection fails!")
                
    def projection_storage(Pmax, Pmin, S, p, q): # charging power is positive
        '''
        Pmax, Pmin from soc limits and current soc; Pmax >= 0; Pmin <= 0
        '''
        if Pmax <0 or Pmin >0:
            return Exception("Check battery soc!")
        if p >= 0:
            if Pmax >= S:
                P = S
            else:
                P = Pmax
            return projection_pv(P, S, p, q)
        else:
            if -Pmin >= S:
                P = S
            else:
                P = -Pmin
            return projection_pv(P, S, -p, q) * np.array([-1, 1])
    
    
    
    # store all iterates
    mat_p_storage = np.zeros((n_timesteps*n_itr, n_storage))
    mat_q_storage = np.zeros((n_timesteps*n_itr, n_storage))
    mat_soc_storage = np.zeros((n_timesteps*n_itr+1, n_storage))
    mat_p_pv = np.zeros((n_timesteps*n_itr, n_pv))
    mat_q_pv = np.zeros((n_timesteps*n_itr, n_pv))
    mat_v = np.ones((n_timesteps*n_itr, n_bus-1))
    mat_loading = np.zeros(n_timesteps*n_itr)
    mat_lambdas = np.zeros((n_timesteps*n_itr, n_bus-1))
    mat_gamma = np.zeros((n_timesteps*n_itr, n_bus-1))
    mat_pi = np.zeros(n_timesteps*n_itr)
    
    # initialize
    p_pv = copy.deepcopy(sgen_p[0,:]) 
    q_pv = np.zeros(n_pv) 
    p_storage = np.zeros(n_storage) 
    q_storage = np.zeros(n_storage) 
    lambdas = np.zeros(n_bus-1)
    gamma = np.zeros(n_bus-1)
    pi = 0.0
    soc = soc_init * np.ones(n_storage)
    mat_soc_storage[0,:] = soc
    
    # OFO iterations
    for itr in tqdm(range(n_timesteps * n_itr)):
        
        if itr % n_itr == 0:
            load_p_current = load_p[itr//n_itr,:]
            load_q_current = load_q[itr//n_itr,:]
            sgen_p_current = sgen_p[itr//n_itr,:]
            p_pv = np.minimum(p_pv, sgen_p_current)
            
        if not pv_control:
            p_pv = copy.deepcopy(sgen_p_current)
            q_pv = np.zeros(n_pv)
        
        # save iterates
        mat_p_storage[itr,:] = p_storage
        mat_q_storage[itr,:] = q_storage
        soc += (eta_ch*p_storage*(p_storage>=0)+p_storage*(p_storage<=0)/eta_dis)*itr_length/E_storage
        mat_soc_storage[itr+1,:] = soc
        mat_p_pv[itr,:] = p_pv
        mat_q_pv[itr,:] = q_pv
    
        # power flow
        if pf == 'pp':
            net.load.p_mw = load_p_current * 1E-3
            net.load.q_mvar = load_q_current * 1E-3
            net.sgen.p_mw = p_pv * 1E-3
            net.sgen.q_mvar = q_pv * 1E-3
            net.storage.p_mw = p_storage * 1E-3
            net.storage.q_mvar = q_storage * 1E-3
            pp.runpp(net)
            v = net.res_bus.vm_pu.to_numpy()[1:]
            loading = net.res_trafo.loading_percent[0]/100.0
            P_trafo = net.res_trafo.p_hv_mw[0] * 1E3
            Q_trafo = net.res_trafo.q_hv_mvar[0] * 1E3
        else:
            dict_return = powerflow(load_p_current, load_q_current, p_pv, q_pv, p_storage, q_storage)
            v = dict_return["v"]
            loading = dict_return["loading"]
            P_trafo = dict_return["P_trafo"]
            Q_trafo = dict_return["Q_trafo"]
        pf_trafo = P_trafo / np.sqrt(P_trafo**2+Q_trafo**2)
        rpf_trafo = Q_trafo / np.sqrt(P_trafo**2+Q_trafo**2)
        mat_v[itr, :] = v
        mat_loading[itr] = loading
    
        
        # dual gradient ascent steps
        lambdas += beta * (v - v_upp)
        gamma += beta * (v_low - v)
        lambdas *= (lambdas >= 0)
        gamma *= (gamma >= 0)
        pi += rho * (loading - 1.0)
        pi *= (pi >= 0)
        
        # primal gradient descent
        if pv_control:
            grad_p_pv = p_pv - sgen_p_current + mat_R_pv.T @ (lambdas - gamma) - pi*pf_trafo/S_trafo*np.ones(n_pv)
            grad_q_pv = xi * q_pv + mat_X_pv.T @ (lambdas - gamma) - pi*rpf_trafo/S_trafo*np.ones(n_pv)
            p_pv -= alpha * grad_p_pv
            q_pv -= alpha * grad_q_pv
            for i in range(n_pv):
                p_pv[i], q_pv[i] = projection_pv(sgen_p_current[i], S_pv[i], p_pv[i], q_pv[i])
        
        grad_p_storage = w2 * p_storage + epsilon * E_storage * (soc-e) + mat_R_storage.T @ (lambdas - gamma) + pi*pf_trafo/S_trafo*np.ones(n_storage)
        grad_q_storage = q_storage + mat_X_storage.T @ (lambdas - gamma) + pi*rpf_trafo/S_trafo*np.ones(n_storage)
        p_storage -= alpha * grad_p_storage
        q_storage -= alpha * grad_q_storage
        Pmax = (soc_max - soc)*E_storage/itr_length/eta_ch
        Pmin = -(soc - soc_min)*E_storage/itr_length*eta_dis
        for i in range(n_storage):
            p_storage[i], q_storage[i] = projection_storage(Pmax[i], Pmin[i], 
                                    S_storage[i], p_storage[i], q_storage[i])
            
    
    
    if True:
        # visualization
        end_step = 2880
        for b in list_bus_visual:
            plt.plot(mat_v[:end_step,b-1], label = f'bus {b}')
        plt.legend()
        plt.show()    
        
        plt.plot(mat_loading[:end_step], label = "trafo")
        plt.legend()
        plt.show()
        
        for i in range(4):
            plt.plot(mat_p_pv[:end_step,i], label = f'pv {i}')
        plt.legend()
        plt.show()    
        
        for i in range(4):
            plt.plot(mat_p_storage[:end_step,i], label = f'storage {i}')
        plt.legend()
        plt.show()    
        
        
        for i in range(4):
            plt.plot(mat_soc_storage[:end_step,i], label = f'storage {i}')
        plt.legend()
        plt.show()
        
        
        
        name = ""
        pd.DataFrame(mat_p_pv).to_csv('mat_p_pv' + name +'.csv')
        pd.DataFrame(mat_v).to_csv('mat_v' + name +'.csv')
        pd.DataFrame(mat_loading).to_csv('mat_loading' + name +'.csv')
        pd.DataFrame(mat_p_storage).to_csv('mat_p_storage' + name +'.csv')
        
    pv_max = np.sum(sgen_p) * 0.25
    pv_gen = np.sum(mat_p_pv)/60
    avv = np.average((mat_v[:,54] - v_upp) * (mat_v[:,54] > v_upp))
    alv = np.average((mat_loading - 1.0) * (mat_loading > 1.0))
    
    return pv_max, pv_gen, avv, alv

if __name__ == "__main__":
    start_time = time.time()
    print(main(epsilon=0.5))
    print("Time is: \n")
    print(time.time() - start_time)







