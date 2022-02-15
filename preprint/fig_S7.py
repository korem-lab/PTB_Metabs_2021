import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from metabs_utils import run_DA, run_correlations, fisher_r_to_z
from get_dfs import get_metabs, get_md, get_v2_aa

def figure_S7():

	metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True)
	md = get_md()
	all_race_conds = list(zip((pd.Series(True, index = md.index), md.race == 1, md.race == 0), ('All', 'AA', 'White')))
	black_cond = [all_race_conds[1]]

	DA_res = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, all_race_conds)
	DA_res = DA_res[DA_res['overall_q'] < 0.1]
	DA_metabs = list(set(DA_res.index))
	metabs = metabs[DA_metabs]

	aa = get_v2_aa()
	md = get_md()
	aa = aa.replace(0, np.nan)
	aa = aa.loc[:, aa.notnull().sum() > 19]

	metabs = metabs.loc[aa.index]

	corrs_all = run_correlations(aa, metabs, 50).set_index(['Microbe', 'Metab'])
	corrs_all['abs(R)'] = np.abs(corrs_all['R'])
	corrs_all = corrs_all[((corrs_all['abs(R)'] > 0.25) & (corrs_all['N'] > 50) & (corrs_all['q'] < 0.1))]
	
	corrs_white = run_correlations(aa.loc[(md.race == 0), :], metabs.loc[(md.race == 0), :], 10).set_index(['Microbe', 'Metab'])
	corrs_black = run_correlations(aa.loc[(md.race == 1), :], metabs.loc[(md.race == 1), :], 10).set_index(['Microbe', 'Metab'])

	corrs_merged = corrs_all.join(corrs_black, rsuffix = '_black')
	corrs_merged = corrs_merged.join(corrs_white, rsuffix='_white')

	fisher_df = corrs_merged.copy()
	fisher_df['corr_diff'] = fisher_df['R_black'] - fisher_df['R_white']
	fisher_df['fisher_rz_p'] = [fisher_r_to_z(a, b, c, d) for a,b,c,d in zip(fisher_df['R_white'], fisher_df['R_black'], fisher_df['N_white'], fisher_df['N_black'])]
	fisher_df['-log(p)'] = -np.log10(fisher_df['fisher_rz_p'])
	fisher_df['same_sign'] = np.sign(fisher_df['R_white']*fisher_df['R_black'])


	plt.figure(figsize=(12,12))
	sns.scatterplot(data=fisher_df.sort_values(by='same_sign'), x='corr_diff', y='-log(p)', 
	                s=120, hue='same_sign', palette=['darkgoldenrod', 'darkslategray'], legend=False)

	plt.axhline(-np.log10(0.05), color='maroon', linewidth=3)
	plt.axvline(0, color='gray', linewidth=3)

	plt.xlim((-0.75, 0.75))
	plt.ylabel("")
	plt.xlabel("")

	plt.gca().xaxis.set_tick_params(width=3, length=10)
	plt.gca().yaxis.set_tick_params(width=3, length=10)

	for axis in ['top','bottom','left','right']:
	    plt.gca().spines[axis].set_linewidth(3)


	plt.gca().set_yticks([0, -np.log10(0.5), -np.log10(0.1), -np.log10(0.05), -np.log10(0.01), -np.log10(0.001)])
	labels = [item.get_text() for item in plt.gca().get_yticklabels()]
	for i in range(len(labels)):
	    labels[i] = ""
	plt.gca().set_yticklabels([1, 0.5, 0.1, 0.05, 0.01, 0.001])

	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)

	plt.savefig("./figurePanels/S7.png", dpi=800, bbox_inches='tight')
	plt.savefig("./figurePanels/S7.pdf", dpi=800, bbox_inches='tight')