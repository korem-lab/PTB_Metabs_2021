import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm

from get_dfs import get_metabs, get_md
from metabs_utils import run_DA


def figures_2AD():

	def make_heatmap(ifn, conds, title, ofn, df=None, y_ticks=True, x_ticks=False, title_on=False, cbar_on=True, cbar_pos='left',  full=True, p_05_to_q=0.5):

		res_all = df
		res_sig = res_all[res_all.heatmap_q < 0.1]
		res_sig = res_sig[res_sig['Cond'].isin(conds)]
		res_all = res_all[res_all.index.get_level_values(0).isin(res_sig.index)]

		q_bins = [.01, .05, .1, p_05_to_q, 1.00001]
		res_all['q_discrete'] = res_all.heatmap_q.apply(lambda p: [b for b in q_bins if p < b][0])
		res_all['slogq'] = res_all.stat.apply(np.sign) * res_all.heatmap_q.apply(np.log10)

		conds = list(res_all.Cond.unique())
		x = res_all.set_index('Cond', append=True).slogq.unstack(1)[conds] ## new
		x = x.assign(m = x.mean(1)).sort_values('m').drop('m', axis = 1)
		sns.set_context('talk')
		x.columns = conds
		sns.set(font="Calibri", font_scale = .9, style = 'whitegrid', context = 'talk')

		color_bins = ['#1d50a3', '#6a7bbb', '#abb1d8', '#e9eaf1', 'white', '#f3e8e2', '#f9896c', '#f76449', '#ff0013']
		cmap = ListedColormap(color_bins[1:-1])#.with_extremes(over = bins[-1], under = bins[1])
		cmap.set_over(color_bins[-1])
		cmap.set_under(color_bins[0])
		bounds = list(np.log10(q_bins[:-1])) + list(-np.log10(q_bins)[:-1][::-1])

		norm = BoundaryNorm(bounds, cmap.N)
		plt.close('all')
		x.index = x.index.to_series().str.capitalize().str.replace('*', '')

		N_metabs = len(set(res_all.index))
		N_conds = len(res_all['Cond'].unique())
		fig = plt.figure(figsize = (N_metabs, 2*N_conds))
		g = sns.heatmap(x[x.columns[::-1]].T, cmap=cmap, norm = norm, yticklabels = y_ticks, xticklabels = x_ticks, cbar=cbar_on, linewidths=2, linecolor='gray')

		##remove colorbar tick labels
		cax = plt.gcf().axes[-1]
		cax.set_yticklabels([])

		_=g.set_xticklabels(g.get_xticklabels(), rotation=60, ha='right', fontsize = min(10, 10*(32/x[x.columns[::-1]].shape[0])), fontweight=10)
			
		if title_on == True:
			plt.title(title)

		height = x[x.columns[::-1]].T.shape[0]
		width = x[x.columns[::-1]].T.shape[1]
		g.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor='gray', lw=3))

		plt.savefig(f"{ofn}.png", dpi = 600, bbox_inches = 'tight')
		plt.savefig(f"{ofn}.pdf", dpi = 600, bbox_inches = 'tight')
		plt.close()



	### Fig 2A - ptb DA heatmap 
	metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True)
	md = get_md().loc[metabs.index]
	all_race_conds = list(zip((pd.Series(True, index = md.index), md.race == 1, md.race == 0), ('All', 'AA', 'White')))
	black_cond = [all_race_conds[1]]

	all_ptb_vs_tb_strict_q = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, all_race_conds)
	all_ptb_vs_tb_strict_q['heatmap_q'] = multipletests(all_ptb_vs_tb_strict_q['p'], method='fdr_bh')[1]
	make_heatmap(None, ['All', 'AA', 'White'], "all sPTB vs TB", "./figurePanels/2A", 
		df=all_ptb_vs_tb_strict_q, y_ticks=False, x_ticks=False, cbar_on=False, p_05_to_q=0.5)
	make_heatmap(None, ['All', 'AA', 'White'], "all sPTB vs TB", "./figurePanels/2A_xticks", 
		df=all_ptb_vs_tb_strict_q, y_ticks=False, x_ticks=True, cbar_on=False, p_05_to_q=0.5)

	### Fig 2D
	extremely_ptb_vs_rest = run_DA(metabs, md, ['extreme_sPTB'], ['very_sPTB', 'moderate_sPTB', 'TB'], False, black_cond)
	extremely_ptb_vs_rest['Cond'] = 'Below 28 (AA) vs 28 and above (AA)'
	very_and_extreme_ptb_vs_rest = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB'], ['moderate_sPTB', 'TB'], False, black_cond)
	very_and_extreme_ptb_vs_rest['Cond'] = 'Below 32 (AA) vs 32 and above (AA)'
	merged = pd.concat([very_and_extreme_ptb_vs_rest, extremely_ptb_vs_rest])
	merged['heatmap_q'] = multipletests(merged['p'], method='fdr_bh')[1]
	make_heatmap(None, list(merged.Cond.unique()), "Above GA thresh vs below GA thresh", "./figurePanels/2D", 
		df=merged, y_ticks=False, x_ticks=False, title_on=False, cbar_on=False, p_05_to_q=0.3)
	make_heatmap(None, list(merged.Cond.unique()), "Above GA thresh vs below GA thresh", "./figurePanels/2D_xticks", 
		df=merged, y_ticks=False, x_ticks=True, title_on=False, cbar_on=False, p_05_to_q=0.3)

