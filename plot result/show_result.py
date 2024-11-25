import copy
import importlib
import data_process_rl as data_init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas
import pandapower as pp
import numpy as np
import random
import collections
from tqdm import tqdm
import time
from pathlib import *
import multiprocessing
import pickle

path_loading = Path.cwd()
path_saving = path_loading / 'result'

#set up gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# create a new data for runing
data_env = importlib.reload(data_init)
# Hyperparameters
lr_mu = 0.0001  # learning rate of actor network，（25）in the paper 0.0001
lr_q = 0.0005  # learning rate of critic network，（24）in the paper 0.001
gamma = 0.99  # discount factor， 0.7 in the paper
batch_size = 32
buffer_limit = 500000
tau = 0.001  # for target soft update （26） （27） in the paper
noise_decrease_factor = 0.993   # noise sigma decreasing factor
n_agent = data_env.n_agent
n_bus = data_env.n_bus
week_num = 4
week_train_num = 3
day_num = 7
quarter_num = data_env.n_quarter
n_itr = data_env.n_itr  # 15
n_pv = data_env.n_pv # 48, 50% nodes have a pv
n_storage = data_env.n_storage
n_load = data_env.n_load
act_dim = data_env.act_dim  # p and q of storage: 2*19=38
s_dim = data_env.s_dim  # SoC, p_PV, p_load, q_PV, q_load, V: 521
soc_init = data_env.soc_init
soc_max = data_env.soc_max
soc_min = data_env.soc_min
eta_ch = data_env.eta_ch
eta_dis = data_env.eta_dis
E_storage = data_env.E_storage
S_storage = data_env.S_storage
itr_length = data_env.itr_length
r_v_scale = 1
r_loading_scale = 1
r_pv_scale = 1
pf = 'pgm'
powerflow = data_init.powerflow


def projection_pv(P, S, p, q):
    '''
    P: active power limit
    S: inverter capacity
    p, q: point to be projected
    Optimal Power Flow Pursuit, appendix B setpoint update
    '''

    Q = np.sqrt(S ** 2 - P ** 2)
    if P == 0.0:
        if q >= S:
            return np.array([0, S])
        elif q <= -S:
            return np.array([0, -S])
        else:
            return np.array([0, q])
    else:
        # region Y
        if (p ** 2 + q ** 2 <= S ** 2) and (p >= 0) and (p <= P):
            return np.array([p, q])
        # region A
        elif (p ** 2 + q ** 2 >= S ** 2) and (p >= 0) and (
                (q >= Q / P * p) or (q <= -Q / P * p)
        ):
            return np.array([p, q]) * S / np.sqrt(p ** 2 + q ** 2)
        # region B
        elif (q >= Q) and (q <= Q / P * p):
            return np.array([P, Q])
        elif (q <= -Q) and (q >= -Q / P * p):
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


def projection_storage(Pmax, Pmin, S, p, q):  # charging power is positive
    '''
    Pmax, Pmin from soc limits and current soc; Pmax >= 0; Pmin <= 0
    '''
    if Pmax < 0 or Pmin > 0:
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


def main(i_seed):
    np.random.seed(i_seed)
    # store result
        # rl
    score_save = np.zeros(1)
    score_v_save = np.zeros(1)
    score_loading_save = np.zeros(1)
    loading_save = np.zeros(n_itr * quarter_num * day_num - 1)
    v_save = np.zeros((n_itr * quarter_num * day_num - 1, n_bus - 1))
    soc_save = np.zeros((n_itr * quarter_num * day_num - 1, n_storage))
        # storage
    p_storage_pd = pandas.read_excel('storage_p_1.xlsx', sheet_name='Sheet1', engine='openpyxl')
    q_storage_pd = pandas.read_excel('storage_q_1.xlsx', sheet_name='Sheet1', engine='openpyxl')
    p_storage_pd = p_storage_pd.drop(columns='Unnamed: 0')
    q_storage_pd = q_storage_pd.drop(columns='Unnamed: 0')
    p_storage_np = p_storage_pd.to_numpy(dtype='float64')
    q_storage_np = q_storage_pd.to_numpy(dtype='float64')
        # pv
    p_pv_pd = pandas.read_excel('pv_p_1.xlsx', sheet_name='Sheet1', engine='openpyxl')
    q_pv_pd = pandas.read_excel('pv_q_1.xlsx', sheet_name='Sheet1', engine='openpyxl')
    p_pv_pd = p_pv_pd.drop(columns='Unnamed: 0')
    q_pv_pd = q_pv_pd.drop(columns='Unnamed: 0')
    p_pv_np = p_pv_pd.to_numpy(dtype='float64')
    q_pv_np = q_pv_pd.to_numpy(dtype='float64')
    S_pv = data_init.S_pv
    ################################## test set
    week = range(week_num)[-1]  # randomly select one week to train
    score = 0
    score_v = 0
    score_loading = 0
    # initial state for rl
    load_p_current = copy.deepcopy(data_init.load_p[week * 7 * 24 * 4, :])
    load_q_current = copy.deepcopy(data_init.load_q[week * 7 * 24 * 4, :])
    sgen_p_current = copy.deepcopy(copy.deepcopy(data_init.sgen_p[week * 7 * 24 * 4, :]))
    p_pv_current = copy.deepcopy(copy.deepcopy(data_init.sgen_p[week * 7 * 24 * 4, :]))
    q_pv_current = np.zeros(n_pv)
    soc_current = copy.deepcopy(soc_init * np.ones(n_storage))
    net = data_init.net
    # power flow
    pp_st = time.time()
    if pf == 'pp':
        net.load.p_mw = load_p_current * 1E-3
        net.load.q_mvar = load_q_current * 1E-3
        net.sgen.p_mw = p_pv_current * 1E-3
        net.sgen.q_mvar = q_pv_current * 1E-3
        net.storage.p_mw = 0 * 1E-3
        net.storage.q_mvar = 0 * 1E-3
        pp.runpp(net)
        # read power flow result
        v_current = net.res_bus.vm_pu.to_numpy()
        loading_current = net.res_trafo.loading_percent[0] / 100.0
        P_trafo = net.res_trafo.p_hv_mw[0] * 1E3
        Q_trafo = net.res_trafo.q_hv_mvar[0] * 1E3
        pf_trafo = P_trafo / np.sqrt(P_trafo ** 2 + Q_trafo ** 2)
        rpf_trafo = Q_trafo / np.sqrt(P_trafo ** 2 + Q_trafo ** 2)
    else:
        dict_return = powerflow(load_p_current, load_q_current, p_pv_current, q_pv_current, np.zeros(n_storage), np.zeros(n_storage))
        v_current = dict_return["v"]
        loading_current = dict_return["loading"]
        P_trafo = dict_return["P_trafo"]
        Q_trafo = dict_return["Q_trafo"]
    pp_et = time.time()
    #  loop for one week
    for i_time in range(n_itr * quarter_num * day_num - 1):  # 15*96*7
        # update state
        if i_time != 0:
            load_p_current = copy.deepcopy(load_p_next)
            load_q_current = copy.deepcopy(load_q_next)
            sgen_p_current = copy.deepcopy(sgen_p_next)
            soc_current = copy.deepcopy(soc_next)
            v_current = copy.deepcopy(v_next)
            loading_current = copy.deepcopy(loading_next)
        p_pv_current = p_pv_np[i_time,:]
        q_pv_current = q_pv_np[i_time,:]
        # rl actions
        # state
        s_soc = copy.deepcopy(soc_current)
        s_p_pv = copy.deepcopy(p_pv_current)
        s_p_load = copy.deepcopy(load_p_current)
        s_q_load = copy.deepcopy(load_q_current)
        s_v = (v_current - 0.95) / (1.05 - 0.95)
        s_loading = copy.deepcopy(np.array([loading_current]))
        s_current = np.concatenate((s_soc, s_p_pv, s_p_load, s_q_load, s_v, s_loading))
        p_storage = p_storage_np[i_time,:]
        q_storage = q_storage_np[i_time,:]
        Pmax = (soc_max - soc_current) * E_storage / itr_length / eta_ch
        Pmin = -(soc_current - soc_min) * E_storage / itr_length * eta_dis
        for i in range(n_storage):
            p_storage[i], q_storage[i] = projection_storage(Pmax[i], Pmin[i],
                                                            S_storage[i], p_storage[i], q_storage[i])

        # project PV action
        p_pv_future = copy.deepcopy(data_init.sgen_p[week * 7 * 24 * 4 + i_time // n_itr, :])
        for i in range(n_pv):
            p_pv_current[i], q_pv_current[i] = projection_pv(sgen_p_current[i], S_pv[i], p_pv_current[i], q_pv_current[i])
            if p_pv_current[i] >= p_pv_future[i]:
                p_pv_current[i] = p_pv_future[i]


        # update next time
        soc_next = copy.deepcopy(soc_current)
        soc_next += (eta_ch * p_storage * (p_storage >= 0) + p_storage * (
                p_storage <= 0) / eta_dis) * itr_length / E_storage

        # power flow
        pp_st = time.time()
        if pf == 'pp':
            net.load.p_mw = load_p_current * 1E-3
            net.load.q_mvar = load_q_current * 1E-3
            net.sgen.p_mw = p_pv_current * 1E-3
            net.sgen.q_mvar = q_pv_current * 1E-3
            net.storage.p_mw = p_storage * 1E-3
            net.storage.q_mvar = q_storage * 1E-3
            pp_st = time.time()
            pp.runpp(net)
            pp_et = time.time()
            v_next = net.res_bus.vm_pu.to_numpy()
            loading_next = net.res_trafo.loading_percent[0] / 100.0
            P_trafo = net.res_trafo.p_hv_mw[0] * 1E3
            Q_trafo = net.res_trafo.q_hv_mvar[0] * 1E3
        else:
            dict_return = powerflow(load_p_current, load_q_current, p_pv_current, q_pv_current, p_storage, q_storage)
            v_next = dict_return["v"]
            loading_next = dict_return["loading"]
            P_trafo = dict_return["P_trafo"]
            Q_trafo = dict_return["Q_trafo"]
        pp_et = time.time()
        # reward
        r_v = 0
        r_loading = 0
        r_pv_p = 0

        for v_value in v_next:
            if v_value < 0.95:
                r_v -= (0.95 - v_value)
            elif v_value > 1.05:
                r_v -= (v_value - 1.05)

        i_count = 0
        for p_pv_value in p_pv_current:
            r_pv_p -= (p_pv_value - p_pv_future[i_count]) ** 2
            i_count += 1

        if 1 < loading_next:
            r_loading -= (loading_next - 1)

        r = r_v_scale * r_v + r_loading_scale * r_loading + r_pv_scale * r_pv_p

        score += r
        score_v += r_v_scale * r_v
        score_loading += r_loading_scale * r_loading

        loading_save[i_time] = loading_next
        v_save[i_time, :] = v_next
        soc_save[i_time, :] = soc_next
        """
        mat_v[itr, :] = v
        mat_loading[itr] = loading
        """

        # update state
        if i_time != 0 and i_time % n_itr == 0:
            load_p_next = data_init.load_p[week * 7 * 24 * 4 + i_time // n_itr, :]
            load_q_next = data_init.load_q[week * 7 * 24 * 4 + i_time // n_itr, :]
            sgen_p_next = data_init.sgen_p[week * 7 * 24 * 4 + i_time // n_itr, :]
        else:
            load_p_next = copy.deepcopy(load_p_current)
            load_q_next = copy.deepcopy(load_q_current)
            sgen_p_next = copy.deepcopy(sgen_p_current)


    # save score
    score_save[0] = score
    score_v_save[0] = score_v
    score_loading_save[0] = score_loading


    ############save result
    score_save_pd = pandas.DataFrame(score_save)
    score_save_pd.to_excel(path_saving / f"score_{i_seed}.xlsx", startcol=0)
    score_v_save_pd = pandas.DataFrame(score_v_save)
    score_v_save_pd.to_excel(path_saving / f"score_v_{i_seed}.xlsx", startcol=0)
    score_loading_save_pd = pandas.DataFrame(score_loading_save)
    score_loading_save_pd.to_excel(path_saving / f"score_loading_{i_seed}.xlsx", startcol=0)
    v_save_pd = pandas.DataFrame(v_save)
    v_save_pd.to_excel(path_saving / f"v_{i_seed}.xlsx", startcol=0)
    loading_save_pd = pandas.DataFrame(loading_save)
    loading_save_pd.to_excel(path_saving / f"loading_{i_seed}.xlsx", startcol=0)
    soc_save_pd = pandas.DataFrame(soc_save)
    soc_save_pd.to_excel(path_saving / f"soc_{i_seed}.xlsx", startcol=0)


if __name__ == '__main__':
    """
    pool = multiprocessing.Pool(processes=5)
    for i_seed in seed_list:
        pool.apply_async(main, (i_seed,))

    pool.close()
    pool.join()
    """
    main(1)