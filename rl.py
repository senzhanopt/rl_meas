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
path_saving = path_loading / 'RL result'

#set up gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# create a new data for runing
data_env = importlib.reload(data_init)
# Hyperparameters
lr_mu = 1.7918800157156533e-05  # learning rate of actor network，（25）in the paper 0.0001
lr_q = 0.003164641782114985  # learning rate of critic network，（24）in the paper 0.001
gamma = 0.7175638451488489  # discount factor， 0.7 in the paper
batch_size = 512
buffer_limit = 500000000000
tau = 0.002855707953807889  # for target soft update （26） （27） in the paper
noise_decrease_factor = 0.912845791926337   # noise sigma decreasing factor
n_agent = data_env.n_agent
week_num = 4
week_train_num = 3
day_num = 7
quarter_num = data_env.n_quarter
n_itr = data_init.n_itr  # 1
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
r_v_scale = 9.050585110542519
r_loading_scale = 24.101745956530074
r_pv_scale = 0.165347515154267
epi_num = 1000
pf = 'pgm'
powerflow = data_init.powerflow


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)  # 创建一个固定长度的队列。当有新纪录加入而队列已满时会自动移除最老的那条记录

    def put(self, transition):
        self.buffer.append(transition)  # 将transition加到buffer中

    def sample(self, n):  # sample n randomly for updating the network
        mini_batch = random.sample(self.buffer, n)  # sample from buffer
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = \
            [], [], [], [], []  # done_mask_lst代表是否为最后一步

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float).to(device),\
               torch.tensor(a_lst).to(device), torch.tensor(r_lst).to(device), \
               torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
               torch.tensor(done_mask_lst).to(device)  # 将array转化为tensor

    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):  # 搭建actor的神经网络，输入state输出action
    def __init__(self):
        super(MuNet, self).__init__()  # 401*1048*256*72
        self.fc1 = nn.Linear(s_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, act_dim)  # action是72维

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用relu激活函数
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))  # tanh的输出在-1,1之间

        return mu  # 输出action


class QNet(nn.Module):  # 搭建critic神经网络，输入state和action，输出Q值
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(s_dim, s_dim)  # s_dim
        self.fc_a = nn.Linear(act_dim, act_dim)  # act_dim
        self.fc_q = nn.Linear(s_dim+act_dim, 16)
        self.fc_3 = nn.Linear(16, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        a = a.to(torch.float32)
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)  # 将h1,h2合并为[h1, h2]
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)  # 未设置输出的激活函数，因为q值范围未知

        return q

class QNet2(nn.Module):  # 搭建critic神经网络，输入state和action，输出Q值
    def __init__(self):
        super(QNet2, self).__init__()
        self.fc_s = nn.Linear(s_dim, s_dim)  # s_dim
        self.fc_a = nn.Linear(act_dim, act_dim)  # act_dim
        self.fc_q = nn.Linear(s_dim+act_dim, 16)
        self.fc_3 = nn.Linear(16, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        a = a.to(torch.float32)
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)  # 将h1,h2合并为[h1, h2]
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)  # 未设置输出的激活函数，因为q值范围未知

        return q


def train(mu, mu_target, q1, q1_target, q2, q2_target, memory, q1_optimizer, q2_optimizer, mu_optimizer, t):
    # 更新q_optimizer和mu_optimizer的参数，用于mu网络和q网络

    s, a, r, s_prime, done = memory.sample(batch_size)  # 从memory中采样batch_size个训练样本
    mu_target_input = mu_target(s_prime).to(device)
    target1 = r + (torch.ones(done.size()).to(device) - done.long().to(device)) * gamma * q1_target(
        s_prime.to(device), mu_target_input)  # 通过Q_target网络计算r+gamma*Q'(s_(n+1),a_(n+1))的值
    target2 = r + (torch.ones(done.size()).to(device) - done.long().to(device)) * gamma * q2_target(
        s_prime.to(device), mu_target_input)  # 通过Q_target网络计算r+gamma*Q'(s_(n+1),a_(n+1))的值
    target1 = target1.to(torch.float32)
    target2 = target2.to(torch.float32)
    target = target1.to(torch.float32)
    for i in range(target.size(dim=0)):
        if target2.data[i] < target1.data[i]:
            target1.data[i] = target2.data[i]
        else:
            target2.data[i] = target1.data[i]

    target1 = target1.to(torch.float32)
    target2 = target2.to(torch.float32)
    q1_value = q1(s, a)
    q2_value = q2(s, a)

    q1_loss = F.smooth_l1_loss(q1(s.to(device), a.to(device)),
                               target1.detach().to(device))  # 计算TD error，Q用smoothl1loss作为损失函数
    q1_optimizer.zero_grad()  # 将网络中的参数设为0
    q1_loss.backward()  # 反向传播
    q1_optimizer.step()  # 更新网络参数

    q2_loss = F.smooth_l1_loss(q2(s.to(device), a.to(device)),
                               target2.detach().to(device))  # 计算TD error，Q用smoothl1loss作为损失函数
    q2_optimizer.zero_grad()  # 将网络中的参数设为0
    q2_loss.backward()  # 反向传播
    q2_optimizer.step()  # 更新网络参数

    # delay
    if (t % (n_itr*96*2) == 0) == 0:
        mu_input = mu(s.to(device))
        mu_loss = -q1(s.to(device), mu_input.to(device)).mean()
        mu_optimizer.zero_grad()
        mu_loss.backward()
        mu_optimizer.step()


def soft_update(net, net_target):  # 更新target网络的参数
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


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
    random.seed(i_seed)
    torch.manual_seed(i_seed)
    sigma = 0.3
    # create variables for rl
    score_history = []
    score = []
    memory = ReplayBuffer()
    q1, q1_target = QNet().to(device), QNet().to(device)  # 创建q，q_target网络
    q2, q2_target = QNet2().to(device), QNet2().to(device)  # 创建q，q_target网络
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    score = 0.0  # 初始化score
    mu, mu_target = MuNet().to(device), MuNet().to(device)
    ou_noise = {}
    q1_optimizer = optim.Adam(q1.parameters(), lr=lr_q)
    q2_optimizer = optim.Adam(q2.parameters(), lr=lr_q)
    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    # store result
        # rl
    score_save = np.zeros(epi_num)
    score_v_save = np.zeros(epi_num)
    score_loading_save = np.zeros(epi_num)
    score_pv_save = np.zeros(epi_num)
        # storage
    p_ess_save = np.zeros((7 * 96 * n_itr, n_storage))
    q_ess_save = np.zeros((7 * 96 * n_itr, n_storage))
    p_pv_save = np.zeros((7 * 96 * n_itr, n_pv))
    q_pv_save = np.zeros((7 * 96 * n_itr, n_pv))
    # state RL
    for n_epi in tqdm(range(epi_num)):
        episode_time = 0
        train_time = 0
        powerflow_time = 0
        st = time.time()
        ################################## training set
        week = random.choice(range(week_num))  # randomly select one week to train
        S_pv = data_init.S_pv
        # initial state for rl
        load_p_current = copy.deepcopy(data_init.load_p[week*7*24*4, :])
        load_q_current = copy.deepcopy(data_init.load_q[week*7*24*4, :])
        sgen_p_current = copy.deepcopy(copy.deepcopy(data_init.sgen_p[week*7*24*4, :]))
        p_pv_current = copy.deepcopy(copy.deepcopy(data_init.sgen_p[week*7*24*4, :]))
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
            dict_return = data_init.powerflow(load_p_current, load_q_current, p_pv_current, q_pv_current, np.zeros(n_storage), np.zeros(n_storage))
            v_current = dict_return["v"]
            loading_current = dict_return["loading"]
            P_trafo = dict_return["P_trafo"]
            Q_trafo = dict_return["Q_trafo"]
        pp_et = time.time()
        powerflow_time += pp_et - pp_st
        #  loop for one week
        for i_time in range(n_itr * quarter_num * day_num - 1):   # 15*96*7
            # update state
            if i_time != 0:
                load_p_current = copy.deepcopy(load_p_next)
                load_q_current = copy.deepcopy(load_q_next)
                sgen_p_current = copy.deepcopy(sgen_p_next)
                p_pv_current = copy.deepcopy(p_pv_next)
                soc_current = copy.deepcopy(soc_next)
                v_current = copy.deepcopy(v_next)
                loading_current = copy.deepcopy(loading_next)

            # rl actions
                # state
            s_soc = copy.deepcopy(soc_current)
            s_p_pv = copy.deepcopy(p_pv_current)
            s_p_load = copy.deepcopy(load_p_current)
            s_q_load = copy.deepcopy(load_q_current)
            s_v = (v_current - 0.95)/(1.05-0.95)
            s_loading = copy.deepcopy(np.array([loading_current]))
            s_current = np.concatenate((s_soc, s_p_pv, s_p_load, s_q_load, s_v, s_loading))
                # reduce noise
            if n_epi > 30:
                sigma = sigma * noise_decrease_factor
                # take actions
            if n_epi >= 30:  # after 5 episodes take actions from DNN
                a = mu(torch.from_numpy(s_current).float().to(device)).to(device)  # action: μ(s),from_numpy(s)将s从numpy转化为torch
                noise_temp = np.random.normal(0, sigma, act_dim)
                a = a.cpu().detach().numpy() + noise_temp  # action加niose
                a = np.clip(a, -1, 1)
            else:  # first 80 episodes are warm-up phase with random actions
                a = np.random.uniform(-1, 1, act_dim)

            # rescale action to storage power: positive value charge, negative value discharge
            p_storage = a[:n_storage] * S_storage
            q_storage = a[n_storage:2*n_storage] * S_storage
            p_pv_current = (a[2*n_storage:2*n_storage+n_pv]+1)/2 * S_pv
            q_pv_current = a[2*n_storage+n_pv:2*n_storage+2*n_pv] * S_pv
            # project SOC action
            Pmax = (soc_max - soc_current) * E_storage / itr_length / eta_ch
            Pmin = -(soc_current - soc_min) * E_storage / itr_length * eta_dis
            for i in range(Pmin.size):
                if Pmin[i] >= 0:
                    Pmin[i] = 0.0

            for i in range(n_storage):
                p_storage[i], q_storage[i] = projection_storage(Pmax[i], Pmin[i],
                                                                S_storage[i], p_storage[i], q_storage[i])
            # project PV action
            p_pv_future = copy.deepcopy(data_init.sgen_p[week * 7 * 24 * 4 + i_time // n_itr, :])
            for i in range(n_pv):
                p_pv_current[i], q_pv_current[i] = projection_pv(sgen_p_current[i], S_pv[i], p_pv_current[i], q_pv_current[i])
                if p_pv_current[i] >= p_pv_future[i]:
                    p_pv_current[i] = p_pv_future[i]

                # rescale the action back to -1, 1
            a_pv_p = p_pv_current / S_pv * 2 - 1
            a_pv_q = q_pv_current / S_pv
            a = np.concatenate((p_storage/S_storage, q_storage/S_storage, a_pv_p, a_pv_q))

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
                powerflow_time += pp_et - pp_st
                v_next = net.res_bus.vm_pu.to_numpy()
                loading_next = net.res_trafo.loading_percent[0]/100.0
                P_trafo = net.res_trafo.p_hv_mw[0] * 1E3
                Q_trafo = net.res_trafo.q_hv_mvar[0] * 1E3
            else:
                dict_return = powerflow(load_p_current, load_q_current, p_pv_current, q_pv_current, p_storage, q_storage)
                v_next = dict_return["v"]
                loading_next = dict_return["loading"]
                P_trafo = dict_return["P_trafo"]
                Q_trafo = dict_return["Q_trafo"]
            pp_et = time.time()
            powerflow_time += pp_et - pp_st
            # reward
            r_v = 0
            r_loading = 0
            r_pv = 0

            for v_value in v_next:
                if 0.97 <= v_value <= 1.03:
                    r_v += 0.01
                elif 1.03 < v_value <= 1.05:
                    r_v += 0
                elif 0.95 <= v_value < 0.97:
                    r_v += 0
                elif v_value < 0.95:
                    r_v -= (0.95 - v_value) * 2
                elif v_value > 1.05:
                    r_v -= (v_value - 1.05) * 2

            i_count = 0
            for p_pv_value in p_pv_current:
                r_pv -= (p_pv_value - p_pv_future[i_count])**2
                i_count += 1

            if 1 < loading_next:
                r_loading -= (loading_next - 1)*10

            r = r_v_scale*r_v + r_loading_scale*r_loading + r_pv_scale*r_pv
            """
            mat_v[itr, :] = v
            mat_loading[itr] = loading
            """

            # update state
            if i_time != 0 and i_time % n_itr == 0:
                load_p_next = data_init.load_p[week*7*24*4 + i_time // n_itr, :]
                load_q_next = data_init.load_q[week*7*24*4 + i_time // n_itr, :]
                sgen_p_next = data_init.sgen_p[week*7*24*4 + i_time // n_itr, :]
                p_pv_next = copy.deepcopy(sgen_p_next)
                q_pv_next = np.zeros(n_pv)
            else:
                load_p_next = copy.deepcopy(load_p_current)
                load_q_next = copy.deepcopy(load_q_current)
                sgen_p_next = copy.deepcopy(sgen_p_current)
                p_pv_next = copy.deepcopy(p_pv_current)
                q_pv_next = copy.deepcopy(q_pv_current)

                # state_next
            s_soc_next = copy.deepcopy(soc_next)
            s_p_pv_next = copy.deepcopy(p_pv_next)
            s_p_load_next = copy.deepcopy(load_p_next)
            s_q_load_next = copy.deepcopy(load_q_next)
            s_v_next = (v_next - 0.95)/(1.05-0.95)
            s_loading_next = copy.deepcopy(np.array([loading_next]))
            s_next = np.concatenate((s_soc_next, s_p_pv_next, s_p_load_next, s_q_load_next, s_v_next, s_loading_next))

            done = (i_time == n_itr * quarter_num * day_num - 1 - 1)
            # save to memory
            memory.put((s_current, a, r, s_next, done))  # 储存近memory中
            train_st = time.time()
            # 当memory size大于1000时，buffer足够大，每天更新一次参数
            if memory.size() >= 2000 and i_time % (n_itr*96) == 0:
                for i in range(7):
                    train(mu, mu_target, q1, q1_target, q2, q2_target, memory, q1_optimizer, q2_optimizer, mu_optimizer, i_time)
                    # delay
                    if i_time % (2*n_itr*96) == 0:
                        soft_update(mu, mu_target)
                        soft_update(q1, q1_target)
                        soft_update(q2, q2_target)
            train_et = time.time()
            train_time += train_et - train_st

        ################################## test set
        week = range(week_num)[-1]  # randomly select one week to train
        score = 0
        score_v = 0
        score_loading = 0
        score_pv = 0
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
        powerflow_time += pp_et - pp_st
        #  loop for one week
        for i_time in range(n_itr * quarter_num * day_num - 1):  # 15*96*7
            # update state
            if i_time != 0:
                load_p_current = copy.deepcopy(load_p_next)
                load_q_current = copy.deepcopy(load_q_next)
                sgen_p_current = copy.deepcopy(sgen_p_next)
                p_pv_current = copy.deepcopy(p_pv_next)
                q_pv_current = copy.deepcopy(np.zeros(n_pv))
                soc_current = copy.deepcopy(soc_next)
                v_current = copy.deepcopy(v_next)
                loading_current = copy.deepcopy(loading_next)

            # rl actions
            # state
            s_soc = copy.deepcopy(soc_current)
            s_p_pv = copy.deepcopy(p_pv_current)
            s_p_load = copy.deepcopy(load_p_current)
            s_q_load = copy.deepcopy(load_q_current)
            s_v = (v_current - 0.95) / (1.05 - 0.95)
            s_loading = copy.deepcopy(np.array([loading_current]))
            s_current = np.concatenate((s_soc, s_p_pv, s_p_load, s_q_load, s_v, s_loading))
            # take actions
            a = mu(torch.from_numpy(s_current).float().to(device)).to(device)  # action: μ(s),from_numpy(s)将s从numpy转化为torch
            a = a.cpu().detach().numpy()

            # rescale action to storage power: positive value charge, negative value discharge
            p_storage = a[:n_storage] * S_storage
            q_storage = a[n_storage:2*n_storage] * S_storage
            p_pv_current = (a[2*n_storage:2*n_storage+n_pv]+1)/2 * S_pv
            q_pv_current = a[2*n_storage+n_pv:2*n_storage+2*n_pv] * S_pv
            # project SOC action
            Pmax = (soc_max - soc_current) * E_storage / itr_length / eta_ch
            Pmin = -(soc_current - soc_min) * E_storage / itr_length * eta_dis
            for i in range(Pmin.size):
                if Pmin[i] >= 0:
                    Pmin[i] = 0.0

            for i in range(n_storage):
                p_storage[i], q_storage[i] = projection_storage(Pmax[i], Pmin[i],
                                                                S_storage[i], p_storage[i], q_storage[i])
            # project PV action
            p_pv_future = copy.deepcopy(data_init.sgen_p[week * 7 * 24 * 4 + i_time // n_itr, :])
            for i in range(n_pv):
                p_pv_current[i], q_pv_current[i] = projection_pv(sgen_p_current[i], S_pv[i], p_pv_current[i], q_pv_current[i])
                if p_pv_current[i] >= p_pv_future[i]:
                    p_pv_current[i] = p_pv_future[i]

                # rescale the action back to -1, 1
            a_pv_p = p_pv_current / S_pv * 2 - 1
            a_pv_q = q_pv_current / S_pv
            a = np.concatenate((p_storage/S_storage, q_storage/S_storage, a_pv_p, a_pv_q))

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
                powerflow_time += pp_et - pp_st
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
            powerflow_time += pp_et - pp_st
            # reward
            r_v = 0
            r_loading = 0
            r_pv = 0

            for v_value in v_next:
                if 0.97 <= v_value <= 1.03:
                    r_v += 0.01
                elif 1.03 < v_value <= 1.05:
                    r_v += 0
                elif 0.95 <= v_value < 0.97:
                    r_v += 0
                elif v_value < 0.95:
                    r_v -= (0.95 - v_value) * 2
                elif v_value > 1.05:
                    r_v -= (v_value - 1.05) * 2

            i_count = 0
            for p_pv_value in p_pv_current:
                r_pv -= (p_pv_value - p_pv_future[i_count]) ** 2
                i_count += 1

            if 1 < loading_next:
                r_loading -= (loading_next - 1) * 10

            r = r_v_scale * r_v + r_loading_scale * r_loading + r_pv_scale * r_pv

            score += r
            score_v += r_v_scale * r_v
            score_loading += r_loading_scale * r_loading
            score_pv += r_pv_scale * r_pv
            """
            mat_v[itr, :] = v
            mat_loading[itr] = loading
            """

            # update state
            if i_time != 0 and i_time % n_itr == 0:
                load_p_next = data_init.load_p[week * 7 * 24 * 4 + i_time // n_itr, :]
                load_q_next = data_init.load_q[week * 7 * 24 * 4 + i_time // n_itr, :]
                sgen_p_next = data_init.sgen_p[week * 7 * 24 * 4 + i_time // n_itr, :]
                p_pv_next = copy.deepcopy(sgen_p_next)
                q_pv_next = np.zeros(n_pv)
            else:
                load_p_next = copy.deepcopy(load_p_current)
                load_q_next = copy.deepcopy(load_q_current)
                sgen_p_next = copy.deepcopy(sgen_p_current)
                p_pv_next = copy.deepcopy(p_pv_current)
                q_pv_next = np.zeros(n_pv)

            # save storage
            if n_epi == epi_num - 1:
                p_ess_save[i_time, :] = p_storage
                q_ess_save[i_time, :] = q_storage
                p_pv_save[i_time, :] = p_pv_current
                q_pv_save[i_time, :] = q_pv_current

        # save score
        score_save[n_epi] = score
        score_v_save[n_epi] = score_v
        score_loading_save[n_epi] = score_loading
        score_pv_save[n_epi] = score_pv


        et = time.time()
        episode_time += et - st
        print("episode:", n_epi,  "score:", score, "score v:", score_v, "score loading:", score_loading, "score pv:", score_pv, "time:", episode_time, "powerflow time:",
              powerflow_time, "train time:", train_time)

    ############save result
    score_save_pd = pandas.DataFrame(score_save)
    score_save_pd.to_excel(path_saving / f"score_{i_seed}.xlsx", startcol=0)
    score_v_save_pd = pandas.DataFrame(score_v_save)
    score_v_save_pd.to_excel(path_saving / f"score_v_{i_seed}.xlsx", startcol=0)
    score_loading_save_pd = pandas.DataFrame(score_loading_save)
    score_loading_save_pd.to_excel(path_saving / f"score_loading_{i_seed}.xlsx", startcol=0)
    score_pv_save_pd = pandas.DataFrame(score_pv_save)
    score_pv_save_pd.to_excel(path_saving / f"score_pv_{i_seed}.xlsx", startcol=0)
    p_ess_save_pd = pandas.DataFrame(p_ess_save)
    p_ess_save_pd.to_excel(path_saving / f"storage_p_{i_seed}.xlsx", startcol=0)
    q_ess_save_pd = pandas.DataFrame(q_ess_save)
    q_ess_save_pd.to_excel(path_saving / f"storage_q_{i_seed}.xlsx", startcol=0)
    p_pv_save_pd = pandas.DataFrame(p_pv_save)
    p_pv_save_pd.to_excel(path_saving / f"pv_p_{i_seed}.xlsx", startcol=0)
    q_pv_save_pd = pandas.DataFrame(q_pv_save)
    q_pv_save_pd.to_excel(path_saving / f"pv_q_{i_seed}.xlsx", startcol=0)


if __name__ == '__main__':
    main(1)