import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from get_dfs import get_metabs, get_md
from metabs_utils import run_DA, msea
from definitions import metabolite_annotations_path


def figure_S5():

	## make directory for msea intermediate outputs
	if not os.path.exists("./data/msea"):
		os.makedirs("./data/msea")

	## Run the permuted differential abundance tests if necessary
	metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True)
	md = get_md()
	all_race_conds = list(zip((pd.Series(True, index = md.index), md.race == 1, md.race == 0), ('All', 'AA', 'White')))
	all_cond = [all_race_conds[0]]
	black_cond = [all_race_conds[1]]
	white_cond = [all_race_conds[2]]

	N_permutations = 10000

	if not os.path.exists("./data/msea/all_37_ps.tsv"):
		all_37_ps = pd.DataFrame()
		for i in range(0, N_permutations):
			temp = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], True, all_cond, seed=i, for_msea=True)
			temp = temp[['p']]
			temp.columns = [i]
			all_37_ps = all_37_ps.join(temp, how='outer')
		all_37_ps.to_csv("./data/msea/all_37_ps.tsv", sep='\t', header=True)

	if not os.path.exists("./data/msea/white_37_ps.tsv"):
		white_37_ps = pd.DataFrame()
		for i in range(0, N_permutations):
			temp = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], True, white_cond, seed=i, for_msea=True)
			temp = temp[['p']]
			temp.columns = [i]
			white_37_ps = white_37_ps.join(temp, how='outer')
		white_37_ps.to_csv("./data/msea/white_37_ps.tsv", sep='\t', header=True)

	if not os.path.exists("./data/msea/black_37_ps.tsv"):
		black_37_ps = pd.DataFrame()
		for i in range(0, N_permutations):
			temp = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], True, black_cond, seed=i, for_msea=True)
			temp = temp[['p']]
			temp.columns = [i]
			black_37_ps = black_37_ps.join(temp, how='outer')
		black_37_ps.to_csv("./data/msea/black_37_ps.tsv", sep='\t', header=True)

	if not os.path.exists("./data/msea/black_32_ps.tsv"):
		black_32_ps = pd.DataFrame()
		for i in range(0, N_permutations):
			temp = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB'], ['moderate_sPTB', 'TB'], True, black_cond, seed=i, for_msea=True)
			temp = temp[['p']]
			temp.columns = [i]
			black_32_ps = black_32_ps.join(temp, how='outer')
		black_32_ps.to_csv("./data/msea/black_32_ps.tsv", sep='\t', header=True)

	if not os.path.exists("./data/msea/black_28_ps.tsv"):
		black_28_ps = pd.DataFrame()
		for i in range(0, N_permutations):
			temp = run_DA(metabs, md, ['extreme_sPTB'], ['very_sPTB', 'moderate_sPTB', 'TB'], True, black_cond, seed=i, for_msea=True)
			temp = temp[['p']]
			temp.columns = [i]
			black_28_ps = black_28_ps.join(temp, how='outer')
		black_28_ps.to_csv("./data/msea/black_28_ps.tsv", sep='\t', header=True)


	## Run MSEAs if necessary
	metabolite_annotations = pd.read_csv(metabolite_annotations_path, index_col=0)
	
	if not os.path.exists("./data/msea/all_37_msea.tsv"):
		real_DA = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, all_cond, for_msea=True)
		all_37_ps = pd.read_csv("./data/msea/all_37_ps.tsv", sep='\t', index_col=0)
		all_37_msea = msea(all_37_ps, real_DA, metabolite_annotations)
		all_37_msea.to_csv("./data/msea/all_37_msea.tsv", sep='\t', index=False)

	if not os.path.exists("./data/msea/white_37_msea.tsv"):
		real_DA = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, white_cond, for_msea=True)
		white_37_ps = pd.read_csv("./data/msea/white_37_ps.tsv", sep='\t', index_col=0)
		white_37_msea = msea(white_37_ps, real_DA, metabolite_annotations)
		white_37_msea.to_csv("./data/msea/white_37_msea.tsv", sep='\t', index=False)

	if not os.path.exists("./data/msea/black_37_msea.tsv"):
		real_DA = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, black_cond, for_msea=True)
		black_37_ps = pd.read_csv("./data/msea/black_37_ps.tsv", sep='\t', index_col=0)
		black_37_msea = msea(black_37_ps, real_DA, metabolite_annotations)
		black_37_msea.to_csv("./data/msea/black_37_msea.tsv", sep='\t', index=False)

	if not os.path.exists("./data/msea/black_32_msea.tsv"):
		real_DA = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB'], ['moderate_sPTB', 'TB'], False, black_cond, for_msea=True)
		black_32_ps = pd.read_csv("./data/msea/black_32_ps.tsv", sep='\t', index_col=0)
		black_32_msea = msea(black_32_ps, real_DA, metabolite_annotations)
		black_32_msea.to_csv("./data/msea/black_32_msea.tsv", sep='\t', index=False)

	if not os.path.exists("./data/msea/black_28_msea.tsv"):
		real_DA = run_DA(metabs, md, ['extreme_sPTB'], ['very_sPTB', 'moderate_sPTB', 'TB'], False, black_cond, for_msea=True)
		black_28_ps = pd.read_csv("./data/msea/black_28_ps.tsv", sep='\t', index_col=0)
		black_28_msea = msea(black_28_ps, real_DA, metabolite_annotations)
		black_28_msea.to_csv("./data/msea/black_28_msea.tsv", sep='\t', index=False)

	all_mspa = pd.read_csv("./data/msea/all_37_msea.tsv", sep='\t')
	white_mspa = pd.read_csv("./data/msea/white_37_msea.tsv", sep='\t')
	AA_mspa = pd.read_csv("./data/msea/black_37_msea.tsv", sep='\t')
	AA_32_mspa = pd.read_csv("./data/msea/black_32_msea.tsv", sep='\t')
	AA_28_mspa = pd.read_csv("./data/msea/black_28_msea.tsv", sep='\t')

	### collect sig results after fdr correction
	all_sig_res = []
	msea_dfs = [('All', all_mspa), ('White', white_mspa), ('AA', AA_mspa), ('AA_32', AA_32_mspa), ('AA_28', AA_28_mspa)]
	for t in msea_dfs:
		df = t[1]
		df['msea'] = t[0]
		for set_type in list(df['set_type'].unique()):
			curr_res = df[df['set_type'] == set_type]
			curr_res['q'] = multipletests(curr_res['p'], method='fdr_bh')[1]
			sig_res = curr_res[curr_res['q'] < 0.1]
			if sig_res.shape[0] > 0:
				all_sig_res.append(sig_res)

	all_sig_res = pd.concat(all_sig_res)
	all_sig_res = all_sig_res[['msea', 'metabolite_set_name', 'p']]
	all_sig_res['p'] = -np.log10(all_sig_res['p'])
	all_sig_res = all_sig_res.set_index(['msea', 'metabolite_set_name'])
	all_sig_res = all_sig_res.unstack()
	all_sig_res.index.name = None
	all_sig_res = all_sig_res.transpose()
	all_sig_res.index = all_sig_res.index.droplevel()
	all_sig_res.index.name = None
	all_sig_res = all_sig_res[['All', 'AA', 'White', 'AA_32', 'AA_28']]
	all_sig_res.columns = ['All', 'African American', 'White', 'African American < 32', 'African American < 28']
	all_sig_res = all_sig_res.drop('Global and overview maps')


	plt.figure(figsize=(8,6))
	sns.heatmap(all_sig_res, cmap='Reds', yticklabels = False, xticklabels = False)

	cbar = plt.gca().collections[0].colorbar
	cbar.set_ticks([2.7, 2.5, 2.3])
	cbar.set_ticklabels(['p = 0.002', 'p = 0.003', 'p = 0.005'])

	plt.savefig(f"./figurePanels/S5.png", dpi = 800, bbox_inches = 'tight')
	plt.savefig(f"./figurePanels/S5.pdf", dpi = 800, bbox_inches = 'tight')