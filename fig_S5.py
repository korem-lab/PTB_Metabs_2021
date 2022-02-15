import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from utils.get_dfs import get_metabs, get_md, get_v2_aa
from utils.metabs_utils import run_DA, run_correlations, fisher_r_to_z


def figure_S5B():

	sig_corrs = pd.read_csv("microbe_metabolite_networks/3A_edges.tsv", sep='\t', index_col=[0,1])

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

	corrs_white = run_correlations(aa.loc[(md.race == 0), :], metabs.loc[(md.race == 0), :], 10).set_index(['Microbe', 'Metab'])
	corrs_black = run_correlations(aa.loc[(md.race == 1), :], metabs.loc[(md.race == 1), :], 10).set_index(['Microbe', 'Metab'])

	corrs_merged = sig_corrs.join(corrs_black, rsuffix = '_black')
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
	plt.savefig("./figurePanels/S5B.png", dpi=800, bbox_inches='tight')
	plt.savefig("./figurePanels/S5B.pdf", dpi=800, bbox_inches='tight')



def figure_S5C():

	metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True)
	metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True)
	md = get_md()
	all_race_conds = list(zip((pd.Series(True, index = md.index), md.race == 1, md.race == 0), ('All', 'AA', 'White')))

	twoA_res = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, all_race_conds)
	twoA_res = twoA_res[twoA_res['overall_q'] < 0.1]
	twoA_metabs = set(twoA_res.index)

	aa = get_v2_aa(metabs=True)
	md = get_md(metabs=True).loc[aa.index]
	aa = aa.replace(0, np.nan)
	aa = aa.loc[:, aa.notnull().sum() > 19]

	curr_metabs_df = metabs.loc[aa.index, twoA_metabs]
	curr_metabs_df = curr_metabs_df.loc[md[md.race == 1].index]
	aa = aa.loc[md[md.race == 1].index]

	corrs_black= run_correlations(aa, curr_metabs_df, 36).set_index(['Microbe', 'Metab'])
	corrs_black['abs(R)'] = np.abs(corrs_black['R'])
	corrs_black = corrs_black[((corrs_black['abs(R)'] > 0.25) & (corrs_black['N'] > 11) & (corrs_black['q'] < 0.1))]
	corrs_black.to_csv("microbe_metabolite_networks/S5C_edges.tsv", index=True, sep='\t')


def figure_S5D():

	metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True)
	metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True)
	md = get_md()
	all_race_conds = list(zip((pd.Series(True, index = md.index), md.race == 1, md.race == 0), ('All', 'AA', 'White')))

	twoA_res = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, all_race_conds)
	twoA_res = twoA_res[twoA_res['overall_q'] < 0.1]
	twoA_metabs = set(twoA_res.index)

	aa = get_v2_aa(metabs=True)
	md = get_md(metabs=True).loc[aa.index]
	aa = aa.replace(0, np.nan)
	aa = aa.loc[:, aa.notnull().sum() > 19]

	curr_metabs_df = metabs.loc[aa.index, twoA_metabs]
	curr_metabs_df = curr_metabs_df.loc[md[md.race == 0].index]
	aa = aa.loc[md[md.race == 0].index]

	corrs_white = run_correlations(aa, curr_metabs_df, 11).set_index(['Microbe', 'Metab'])
	corrs_white['abs(R)'] = np.abs(corrs_white['R'])
	corrs_white = corrs_white[((corrs_white['abs(R)'] > 0.25) & (corrs_white['N'] > 11) & (corrs_white['q'] < 0.1))]
	corrs_white.to_csv("microbe_metabolite_networks/S5D_edges.tsv", index=True, sep='\t')



def figure_S5E():
	metabs = get_metabs(min_pres = 0, min_impute=False)
	md = get_md()
	all_race_conds = list(zip((pd.Series(True, index = md.index), md.race == 1, md.race == 0), ('All', 'AA', 'White')))
	black_cond = [all_race_conds[1]]

	twoA_res = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, all_race_conds)
	twoA_res = twoA_res[twoA_res['overall_q'] < 0.1]
	twoA_metabs = set(twoA_res.index)

	extremely_ptb_vs_rest = run_DA(metabs, md, ['extreme_sPTB'], ['very_sPTB', 'moderate_sPTB', 'TB'], False, black_cond)
	extremely_ptb_vs_rest['Cond'] = 'Below 28 (AA) vs 28 and above (AA)'
	very_and_extreme_ptb_vs_rest = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB'], ['moderate_sPTB', 'TB'], False, black_cond)
	very_and_extreme_ptb_vs_rest['Cond'] = 'Below 32 (AA) vs 32 and above (AA)'
	twoD_res = pd.concat([very_and_extreme_ptb_vs_rest, extremely_ptb_vs_rest])
	twoD_res['heatmap_q'] = multipletests(twoD_res['p'], method='fdr_bh')[1]
	twoD_res = twoD_res[twoD_res['heatmap_q'] < 0.1]

	curr_metabs = set(twoD_res.index).difference(twoA_metabs)
	
	aa = get_v2_aa(metabs=True)
	md = get_md(metabs=True).loc[aa.index]
	aa = aa.replace(0, np.nan)
	aa = aa.loc[:, aa.notnull().sum() > 19]

	curr_metabs_df = metabs.loc[aa.index, curr_metabs]
	curr_metabs_df = curr_metabs_df.loc[md[md.race == 1].index]
	aa = aa.loc[md[md.race == 1].index]

	corrs = run_correlations(aa, curr_metabs_df, 37).set_index(['Microbe', 'Metab'])
	corrs['abs(R)'] = np.abs(corrs['R'])
	sig_corrs = corrs[((corrs['abs(R)'] > 0.25) & (corrs['N'] > 50) & (corrs['q'] < 0.1))]
	sig_corrs['sign'] = np.sign(sig_corrs['R'])
	sig_corrs.to_csv("microbe_metabolite_networks/S5E_edges.tsv", sep='\t', index=True)


def figure_S5F():

	sig_corrs = pd.read_csv("microbe_metabolite_networks/3A_edges.tsv", sep='\t', index_col=[0,1])

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

	early_cond = ((md.race == 1) & (md.delGA < 32))
	rest_cond = ((md.race == 1) & (md.delGA >= 32))

	corrs_black_early = run_correlations(aa.loc[early_cond, :], metabs.loc[early_cond, :], 10).set_index(['Microbe', 'Metab'])
	corrs_rest = run_correlations(aa.loc[rest_cond, :], metabs.loc[rest_cond, :], 10).set_index(['Microbe', 'Metab'])

	corrs_merged = sig_corrs.join(corrs_black_early, rsuffix = '_black_early')
	corrs_merged = corrs_merged.join(corrs_rest, rsuffix='_rest')

	fisher_df = corrs_merged.copy()
	fisher_df['corr_diff'] = fisher_df['R_black_early'] - fisher_df['R_rest']
	fisher_df['fisher_rz_p'] = [fisher_r_to_z(a, b, c, d) for a,b,c,d in zip(fisher_df['R_rest'], fisher_df['R_black_early'], fisher_df['N_black_early'], fisher_df['N_rest'])]
	fisher_df['-log(p)'] = -np.log10(fisher_df['fisher_rz_p'])
	fisher_df['same_sign'] = np.sign(fisher_df['R_rest']*fisher_df['R_black_early'])

	plt.figure(figsize=(12,12))
	sns.scatterplot(data=fisher_df.dropna().sort_values(by='same_sign'), x='corr_diff', y='-log(p)', 
					s=120, hue='same_sign', palette={0:'darkgoldenrod', 1:'darkslategray'}, legend=False)

	plt.axhline(-np.log10(0.05), color='maroon', linewidth=3)
	plt.axvline(0, color='gray', linewidth=3)
	plt.xlim((-0.75, 0.75))
	plt.ylim((-.2, 2))
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

	plt.savefig("./figurePanels/S5F.pdf", dpi=600, bbox_inches='tight')
