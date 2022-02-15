import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from metabs_utils import run_DA, run_correlations, fisher_r_to_z
from get_dfs import get_metabs, get_md, get_v2_aa


def figure_3A():

	metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True)
	md = get_md()
	all_race_conds = list(zip((pd.Series(True, index = md.index), md.race == 1, md.race == 0), ('All', 'AA', 'White')))
	black_cond = [all_race_conds[1]]

	DA_res = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, all_race_conds)
	DA_res = DA_res[DA_res['overall_q'] < 0.1]
	DA_metabs = list(set(DA_res.index))
	metabs = metabs[DA_metabs]

	aa = get_v2_aa(metabs=True)
	md = get_md(metabs=True)
	aa = aa.replace(0, np.nan)
	aa = aa.loc[:, aa.notnull().sum() > 19]

	metabs = metabs.loc[aa.index]

	corrs_all = run_correlations(aa, metabs, 50).set_index(['Microbe', 'Metab'])
	corrs_all['abs(R)'] = np.abs(corrs_all['R'])
	corrs_all = corrs_all[((corrs_all['abs(R)'] > 0.25) & (corrs_all['N'] > 50) & (corrs_all['q'] < 0.1))]

	corrs_all.to_csv("./figurePanels/3A_edges.tsv", sep='\t')

