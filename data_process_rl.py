from data_process import *
#################################################### create data for rl
# numbers
n_week = 4
n_week_train = 3
n_day = 7
n_quarter = 96
n_agent = n_storage
n_load = load_p.shape[1]
# diats
day_diats = [i for i in range(n_day)]
agent_diats = [i for i in range(1, +1)]  # [1,..., 10] the index means the node index(0, ..., 10) of the agentday_diats = [i for i in range(day_num)]
week_diats = [i for i in range(n_week)]
week_train_diats = week_diats[0:3]  # [0, 1, 2]
week_test_diats = week_diats[-1]  # [3]
# action and state
act_dim = 2*n_storage  # p and q of storage
s_dim = n_storage + n_pv + 2*n_load + (n_bus-1) + 1  # SoC, p_PV, p_load, q_load, V, loading





