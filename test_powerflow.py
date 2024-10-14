from func_powerflow import pf
import numpy as np


vm = pf(np.ones(95),np.ones(95),np.ones(95),np.ones(95))
print(vm)