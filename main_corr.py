import random
import collections
import numpy as np
import statistics
import random
import pandapower as pp
import pandas as pd
from pandapower.plotting import simple_plot
from pathlib import *
import pickle
import dcor
from scipy import stats


path = Path(__file__).resolve().parent

def main():
    # load data
    data1 = pd.read_excel(path / 'data.xlsx', sheet_name='Sheet1')
    data2 = pd.read_excel(path / 'data.xlsx', sheet_name='Sheet2')

    n_agent = data1.shape[0]
    n_day = data1.shape[1]

    # calculate the correlations
        # spearman
    spearman_corr_day = np.zeros((n_day))
    spearman_corr_agent = np.zeros((n_agent))
    for i_agent in range(n_agent):
        spearman_corr_agent[i_agent] = stats.spearmanr(data1.iloc[i_agent].values, data2.iloc[i_agent].values)[0]

    for i_day in range(n_day):
        spearman_corr_day[i_day] = stats.spearmanr(data1[i_day+1].values, data2[i_day+1].values)[0]

        # Kendall’s tau
    kendall_corr_day = np.zeros((n_day))
    kendall_corr_agent = np.zeros((n_agent))

    for i_agent in range(n_agent):
        kendall_corr_agent[i_agent] = stats.kendalltau(data1.iloc[i_agent].values, data2.iloc[i_agent].values)[0]

    for i_day in range(n_day):
        kendall_corr_day[i_day] = stats.kendalltau(data1[i_day+1].values, data2[i_day+1].values)[0]

        # weighted Kendall’s tau
    w_kendall_corr_day = np.zeros((n_day))
    w_kendall_corr_agent = np.zeros((n_agent))

    for i_agent in range(n_agent):
        w_kendall_corr_agent[i_agent] = stats.weightedtau(data1.iloc[i_agent].values, data2.iloc[i_agent].values)[0]

    for i_day in range(n_day):
        w_kendall_corr_day[i_day] = stats.weightedtau(data1[i_day+1].values, data2[i_day+1].values)[0]

    # save result
        # spearman
    spearman_corr_agent_save = pd.DataFrame(spearman_corr_agent)
    spearman_corr_agent_save.to_excel(path / f"spearman_corr_agent.xlsx", startcol=0, sheet_name='Sheet1')

    spearman_corr_day_save = pd.DataFrame(spearman_corr_day)
    spearman_corr_day_save.to_excel(path / f"spearman_corr_day.xlsx", startcol=0, sheet_name='Sheet1')
        # Kendall’s tau
    kendall_corr_agent_save = pd.DataFrame(kendall_corr_agent)
    kendall_corr_agent_save.to_excel(path / f"kendall_corr_agent.xlsx", startcol=0, sheet_name='Sheet1')

    kendall_corr_day_save = pd.DataFrame(kendall_corr_day)
    kendall_corr_day_save.to_excel(path / f"kendall_corr_day.xlsx", startcol=0, sheet_name='Sheet1')
        # weighted Kendall’s tau
    w_kendall_corr_agent_save = pd.DataFrame(w_kendall_corr_agent)
    w_kendall_corr_agent_save.to_excel(path / f"w_kendall_corr_agent.xlsx", startcol=0, sheet_name='Sheet1')

    w_kendall_corr_day_save = pd.DataFrame(w_kendall_corr_day)
    w_kendall_corr_day_save.to_excel(path / f"w_kendall_corr_day.xlsx", startcol=0, sheet_name='Sheet1')


if __name__ == '__main__':
    main()

