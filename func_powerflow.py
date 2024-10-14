from power_grid_model import PowerGridModel,  initialize_array, LoadGenType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(['science'])


nw_lv_rural2_cable = pd.read_excel("lv_rural2.xlsx", sheet_name="cable")
n_bus = 96
idx_ext = 62
max_i_amper = 270
v_lv = 0.4 # kV
phases = [[i,i%3] for i in range(n_bus) if i != idx_ext] # if unbalanced
balanced = 0

def pgm():
    
    '''
    
    Returns
    -------
    asym_pgm : A PGM object.
    asym_input_data : Dictionary.

    '''
    # same type for all cables
    c_f_per_km = 1E-10
    r_ohm_per_km = 0.2067
    x_ohm_per_km = 0.080425
    c0_f_per_km = 1E-10
    r0_ohm_per_km = 0.2067*4
    x0_ohm_per_km = 0.080425*4
    
    # node
    node = initialize_array("input", "node", n_bus)
    node["id"] = [i for i in range(n_bus)]
    node["u_rated"] = [v_lv*1E3] * n_bus
    # line
    line = initialize_array("input", "line", n_bus-1)
    line["id"] = [i for i in range(n_bus,n_bus*2-1)]
    line["from_node"] = nw_lv_rural2_cable.from_bus.to_numpy()
    line["to_node"] = nw_lv_rural2_cable.to_bus.to_numpy()
    line["from_status"] = [1] * (n_bus-1)
    line["to_status"] = [1] * (n_bus-1)
    line["r1"] = nw_lv_rural2_cable.length_km.to_numpy()*r_ohm_per_km
    line["x1"] = nw_lv_rural2_cable.length_km.to_numpy()*x_ohm_per_km
    line["c1"] = nw_lv_rural2_cable.length_km.to_numpy()*c_f_per_km
    line["tan1"] = [0.0] * (n_bus-1)
    line["r0"] = nw_lv_rural2_cable.length_km.to_numpy()*r0_ohm_per_km
    line["x0"] = nw_lv_rural2_cable.length_km.to_numpy()*x0_ohm_per_km
    line["c0"] = nw_lv_rural2_cable.length_km.to_numpy()*c0_f_per_km
    line["tan0"] = [0.0] * (n_bus-1)
    line["i_n"] = [max_i_amper] * (n_bus-1)
    # source
    source = initialize_array("input", "source", 1)
    source["id"] = [2*n_bus-1]
    source["node"] = [idx_ext]
    source["status"] = [1]
    source["u_ref"] = [1.0]
    source["sk"] = 1E12
    # load
    asym_load = initialize_array("input", "asym_load", n_bus-1)
    asym_load["id"] = [i for i in range(2*n_bus, 3*n_bus-1)]
    asym_load["node"] = [i for i in range(n_bus) if i != idx_ext]
    asym_load["status"] = [1] * (n_bus-1)
    asym_load["type"] = [LoadGenType.const_power]*(n_bus-1)
    asym_load["p_specified"] = np.ones((n_bus-1,3))*1E3
    asym_load["q_specified"] = np.ones((n_bus-1,3))*1E3
    # gen
    asym_gen = initialize_array("input", "asym_gen", n_bus-1)
    asym_gen["id"] = [i for i in range(3*n_bus, 4*n_bus-1)]
    asym_gen["node"] = [i for i in range(n_bus) if i != idx_ext]
    asym_gen["status"] = [1] * (n_bus-1)
    asym_gen["type"] = [LoadGenType.const_power]*(n_bus-1)
    asym_gen["p_specified"] = np.ones((n_bus-1,3))*1E3
    asym_gen["q_specified"] = np.ones((n_bus-1,3))*1E3
    # all
    asym_input_data = {
        "node": node,
        "line": line,
        "asym_load": asym_load,
        "asym_gen": asym_gen,
        "source": source
    }
    asym_pgm = PowerGridModel(input_data=asym_input_data)
    return asym_pgm, asym_input_data

asym_pgm, asym_input_data = pgm()

load_update = initialize_array("update", "asym_load", n_bus-1)
load_update["id"] = asym_input_data['asym_load']['id']
sgen_update = initialize_array("update", "asym_gen", n_bus-1)
sgen_update["id"] = asym_input_data['asym_gen']['id']
update_data = {"asym_load": load_update, "asym_gen": sgen_update}    

def pf(p, q, load_p, load_q): # unit in kW/kVar

    if balanced:
        p_gen_temp = np.repeat(np.reshape(p,[-1,1]),3,axis=1) / 3
        q_gen_temp = np.repeat(np.reshape(q,[-1,1]),3,axis=1) / 3
        p_load_temp = np.repeat(np.reshape(load_p,[-1,1]),3,axis=1) / 3
        q_load_temp = np.repeat(np.reshape(load_q,[-1,1]),3,axis=1) / 3
    else:
        p_gen_temp = np.zeros((n_bus-1,3))
        q_gen_temp = np.zeros((n_bus-1,3))
        p_load_temp = np.zeros((n_bus-1,3))
        q_load_temp = np.zeros((n_bus-1,3))
        for i,em in enumerate(phases):
            p_gen_temp[i,em[1]] = p[i]
            q_gen_temp[i,em[1]] = q[i]
            p_load_temp[i,em[1]] = load_p[i]
            q_load_temp[i,em[1]] = load_q[i]
    
    load_update["p_specified"] = 1E3*p_load_temp
    load_update["q_specified"] = 1E3*q_load_temp
    sgen_update["p_specified"] = 1E3*p_gen_temp
    sgen_update["q_specified"] = 1E3*q_gen_temp
    asym_pgm.update(update_data=update_data)
    if balanced:
        output_data = asym_pgm.calculate_power_flow(symmetric = True)
        vm = output_data["node"]["u_pu"]
        vm = np.delete(vm, idx_ext)
    else:
        output_data = asym_pgm.calculate_power_flow(symmetric = False)
        vm = output_data["node"]["u_pu"]
        vm = np.array([vm[em[0],em[1]] for em in phases])
    return vm