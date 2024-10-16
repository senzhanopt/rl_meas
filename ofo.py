from data_process import *

# online feedback optimization
mat_p_storage = np.zeros((n_timesteps*n_itr, n_storage))
mat_q_storage = np.zeros((n_timesteps*n_itr, n_storage))
mat_soc_storage = np.zeros((n_timesteps*n_itr, n_storage))
mat_p_pv = np.zeros((n_timesteps*n_itr, n_pv))
mat_q_pv = np.zeros((n_timesteps*n_itr, n_pv))
mat_v = np.ones((n_timesteps*n_itr, n_bus-1))
mat_lambdas = np.ones((n_timesteps*n_itr, n_bus-1))
mat_gamma = np.ones((n_timesteps*n_itr, n_bus-1))
