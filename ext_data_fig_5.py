import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from statsmodels.stats.multitest import multipletests

from utils.definitions import ptb_colors
from utils.get_dfs import get_metabs, get_md, get_metab_annotations, get_vp
from utils.metabs_utils import run_DA, make_heatmap, msea


## boxplots of figure 2A metabolites
def ext_data_fig_5A():

	metabs = get_metabs(min_impute=False)
	md = get_md()
	md.race = md.race.map({0:'White\nwomen', 1:'Black\nwomen'})

	data = metabs.join(md)

	curr_metabs = ['diethanolamine', 'choline', 'ethyl beta-glucopyranoside', 
					'tartarate', 'glycerophosphoserine*',
					'spermine', 'glutamate, gamma-methyl ester', '(R)-3-hydroxybutyrylcarnitine']

	for m in curr_metabs:
		fig, axs = plt.subplots(1, 1, figsize = (2, 3.5), sharey=False)
		sns.boxplot(data=data[data.race != 2], x = 'race', y = m, hue='PTB', palette=ptb_colors, showfliers=False, ax = axs)
		sns.swarmplot(data=data[data.race != 2], x = 'race', y = m, hue='PTB', palette=['gray', 'gray'] , size = 2.2, ax = axs, dodge=True, alpha=1)
		axs.set_xlabel('')
		axs.get_legend().remove()
		axs.set_ylabel("")
		axs.set_xticks([])
		ymin, ymax = plt.gca().get_ylim()
		plt.ylim(ymin, ymax*1.15)
		plt.savefig(f"./figurePanels/ext_data_5A_{m}.pdf", dpi=800, bbox_inches='tight')


## KDE plots of xenobiotic levels across samples
def ext_data_fig_5B():

	metabs = get_metabs(min_impute=False, rzscore=True)
	metabs = metabs[['diethanolamine', 'EDTA', 'ethyl beta-glucopyranoside', 'tartarate']]

	f, ax = plt.subplots(figsize=(8, 4))

	x = metabs['diethanolamine'].dropna()
	sns.distplot(x, hist=False, label=f'DEA, N={x.shape[0]}')
	x = metabs['ethyl beta-glucopyranoside'].dropna()
	sns.distplot(x, hist=False, label=f'Ethyl glucoside, N={x.shape[0]}')
	x = metabs['tartarate'].dropna()
	sns.distplot(x, hist=False, label=f'Tartrate, N={x.shape[0]}')
	x = metabs['EDTA'].dropna()
	sns.distplot(x, hist=False, label=f'EDTA, N={x.shape[0]}')

	plt.legend(loc='upper left')
	plt.xlabel("")
	plt.ylabel("")
	plt.savefig(f"./figurePanels/ext_data_5B.pdf", dpi=800, bbox_inches='tight')


## Figure 2A heatmap minus progesterone treated women
def ext_data_fig_5C():

	metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True)
	md = get_md().loc[metabs.index]
	vp = get_vp()
	md = md.join(vp)
	md = md[md.vag_p == 'No']
	metabs = metabs.loc[md.index]
	all_race_conds = list(zip((pd.Series(True, index = md.index), md.race == 1, md.race == 0), ('All', 'AA', 'White')))
	black_cond = [all_race_conds[1]]

	res = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, all_race_conds)
	res['heatmap_q'] = multipletests(res['p'], method='fdr_bh')[1]
	sig_hits = res[res['heatmap_q'] < 0.1]
	sig_hits.to_csv("differential_abundance_sig_hits/ext_data_5C_sig_hits.csv")
	make_heatmap(None, ['All', 'AA', 'White'], "all sPTB vs TB", "./figurePanels/ext_data_5C",
		df=res, y_ticks=False, x_ticks=False, cbar_on=False, p_05_to_q=0.5)
	make_heatmap(None, ['All', 'AA', 'White'], "all sPTB vs TB", "./figurePanels/ext_data_5C_xticks",
		df=res, y_ticks=False, x_ticks=True, cbar_on=False, p_05_to_q=0.5)


## MSEA heatmap
def ext_data_fig_5D():

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
	metabolite_annotations = get_metab_annotations()
	
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
	print(all_sig_res)
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

	plt.savefig(f"./figurePanels/ext_data_5D.png", dpi = 800, bbox_inches = 'tight')
	plt.savefig(f"./figurePanels/ext_data_5D.pdf", dpi = 800, bbox_inches = 'tight')



## Xenobiotic levels in negative controls
def ext_data_fig_5E():
	
	metabs = get_metabs(rzscore=False, min_impute=False)

	### DEA
	fig, ax = plt.subplots(1, 1, figsize=(3,7))

	temp = metabs[['diethanolamine']]
	temp['x'] = 1
	sns.boxplot(data=temp, x='x', y='diethanolamine', ax=ax, palette=['tab:red'])
	sns.swarmplot(data=temp, x='x', y='diethanolamine', dodge=True, ax=ax, palette=['gray', 'gray'], s=2.7)

	neg_mean = np.array([397635])
	neg_min = np.array([178708])
	neg_max =np.array([947403])
	ax.errorbar([1], neg_mean, [neg_mean - neg_min, neg_max - neg_mean], fmt='ok', lw=3, ecolor='red')
	ax.set_yscale('symlog')
	xmin, xmax = ax.get_xlim()
	print(xmin, xmax)
	ax.set_xlim(-0.6,1.4)
	ax.xaxis.set_ticks([0,1])
	ax.xaxis.set_ticklabels(["", ""])
	ax.set_xlabel("")
	ax.set_ylabel("")
	ax.tick_params(axis='y', which='major', labelsize=14)
	plt.savefig("./figurePanels/ext_data_5E_diethanolamine.pdf", dpi=800, bbox_inches='tight')

	### Ethyl glucoside
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3,7), gridspec_kw={'height_ratios': [4,1]})
	fig.subplots_adjust(hspace=0.1)  # adjust space between axes
	temp = metabs[['ethyl beta-glucopyranoside']]
	temp['x'] = 1
	sns.boxplot(data=temp, x='x', y='ethyl beta-glucopyranoside', ax=ax1, palette=['tab:red'])
	sns.swarmplot(data=temp, x='x', y='ethyl beta-glucopyranoside', dodge=True, ax=ax1, palette=['gray', 'gray', 'gray'], s=2.7)
	ax1.set_yscale('symlog')
	ymin, ymax = ax1.get_ylim()
	ax1.set_ylim(50000, ymax)
	ax1.set_xlabel("")
	ax1.set_ylabel("")
	ax1.tick_params(bottom=False, axis='x', which='both') 

	neg_mean = np.array([0])
	neg_min = np.array([0])
	neg_max =np.array([0])
	ax2.errorbar([1], neg_mean, [neg_mean - neg_min, neg_max - neg_mean], fmt='ok', lw=3, ecolor='red')
	ax2.set_xlim(-0.6,1.4)
	ax2.set_ylim(-0.5, 1)  
	ax2.xaxis.set_ticks([0,1])
	ax2.xaxis.set_ticklabels(["", ""])

	## create break
	ax1.spines['bottom'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax1.tick_params(labeltop=False)  # don't put tick labels at the top
	ax2.xaxis.tick_bottom()
	ax2.yaxis.set_ticks([0])
	ax2.yaxis.set_ticklabels([0])
	ax1.tick_params(axis='y', which='major', labelsize=14)
	ax2.tick_params(axis='y', which='major', labelsize=14)
	plt.savefig("./figurePanels/ext_data_5E_ethyl_glucoside.pdf", dpi=800, bbox_inches='tight')


	## tartrate
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3,7), gridspec_kw={'height_ratios': [4,1]})
	fig.subplots_adjust(hspace=0.1)  # adjust space between axes
	temp = metabs[['tartarate']]
	temp['x'] = 1
	sns.boxplot(data=temp, x='x', y='tartarate', ax=ax1, palette=['tab:red'])
	sns.swarmplot(data=temp, x='x', y='tartarate', dodge=True, ax=ax1, palette=['gray', 'gray', 'gray'], s=2.7)
	ax1.set_yscale('symlog')
	ymin, ymax = ax1.get_ylim()
	ax1.set_ylim(100000, ymax)
	ax1.set_xlabel("")
	ax1.set_ylabel("")
	ax1.tick_params(bottom=False, axis='x', which='both') 

	neg_mean = np.array([0])
	neg_min = np.array([0])
	neg_max =np.array([0])
	ax2.errorbar([1], neg_mean, [neg_mean - neg_min, neg_max - neg_mean], fmt='ok', lw=3, ecolor='red')
	ax2.set_xlim(-0.6,1.4)
	ax2.set_ylim(-0.5, 1)  # most of the data
	ax2.xaxis.set_ticks([0,1])
	ax2.xaxis.set_ticklabels(["", ""])

	## create break
	ax1.spines['bottom'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax1.tick_params(labeltop=False)  # don't put tick labels at the top
	ax2.xaxis.tick_bottom()
	ax2.yaxis.set_ticks([0])
	ax2.yaxis.set_ticklabels([0])
	ax1.tick_params(axis='y', which='major', labelsize=14)
	ax2.tick_params(axis='y', which='major', labelsize=14)
	plt.savefig("./figurePanels/ext_data_5E_tartrate.pdf", dpi=800, bbox_inches='tight')


	## EDTA
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3,7), gridspec_kw={'height_ratios': [4,1]})
	fig.subplots_adjust(hspace=0.1)  # adjust space between axes
	temp = metabs[['EDTA']]
	temp['x'] = 1
	sns.boxplot(data=temp, x='x', y='EDTA', ax=ax1, palette=['tab:red'])
	sns.swarmplot(data=temp, x='x', y='EDTA', dodge=True, ax=ax1, palette=['gray', 'gray', 'gray'], s=2.7)
	ax1.set_yscale('symlog')
	# ax1.get_legend().remove()
	ymin, ymax = ax1.get_ylim()
	ax1.set_ylim(20000000, ymax)
	ax1.set_yticks([100000000, 1000000000])
	ax1.set_xlabel("")
	ax1.set_ylabel("")
	ax1.tick_params(bottom=False, axis='x', which='both') 

	neg_mean = np.array([0])
	neg_min = np.array([0])
	neg_max =np.array([0])
	ax2.errorbar([1], neg_mean, [neg_mean - neg_min, neg_max - neg_mean], fmt='ok', lw=3, ecolor='red')
	ax2.set_xlim(-0.6,1.4)
	ax2.set_ylim(-0.5, 1) 
	ax2.xaxis.set_ticks([0,1])
	ax2.xaxis.set_ticklabels(["", ""])

	## create break
	ax1.spines['bottom'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax1.tick_params(labeltop=False)  # don't put tick labels at the top
	ax2.xaxis.tick_bottom()
	ax2.yaxis.set_ticks([0])
	ax2.yaxis.set_ticklabels([0])
	ax1.tick_params(axis='y', which='major', labelsize=14)
	ax2.tick_params(axis='y', which='major', labelsize=14)
	plt.savefig("./figurePanels/ext_data_5E_EDTA.pdf", dpi=800, bbox_inches='tight')

## xenobiotic mass errors
def ext_data_fig_5F():

	xeno_mass_errors = [0.034354, 0.471315, 0.434599, 0.26844] ## from supp table 14
	non_xeno_avg_mass_error = [0.85975]

	plt.figure(figsize=(3, 6))
	plt.scatter(np.zeros(len(xeno_mass_errors)), xeno_mass_errors, color='tab:blue')
	plt.scatter(np.zeros(len(non_xeno_avg_mass_error)), non_xeno_avg_mass_error, color='tab:red')
	# plt.ylabel("Mass error (ppm)", size=14)
	plt.xlabel("")
	plt.xticks([], [])
	plt.yticks(fontsize=14)
	plt.xlim((-0.15, 1))
	plt.savefig("./figurePanels/ext_data_5F.pdf", dpi=800, bbox_inches='tight')
	plt.savefig("./figurePanels/ext_data_5F.png", dpi=800, bbox_inches='tight')
