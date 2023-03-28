import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from utils.definitions import ptb_colors
from utils.get_dfs import get_metabs, get_md
from utils.metabs_utils import run_DA, make_heatmap


def figure_2A():

	metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True, named_only=True)
	md = get_md().loc[metabs.index]
	metabs = metabs.loc[md.index]
	all_race_conds = list(zip((pd.Series(True, index = md.index), md.race == 1, md.race == 0), ('All', 'AA', 'White')))

	res = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, all_race_conds)
	res['heatmap_q'] = multipletests(res['p'], method='fdr_bh')[1]
	sig_hits = res[res['heatmap_q'] < 0.1]
	sig_hits.to_csv("differential_abundance_sig_hits/2A_sig_hits.csv")
	make_heatmap(None, ['All', 'AA', 'White'], "all sPTB vs TB", "./figurePanels/2A", 
		df=res, y_ticks=False, x_ticks=False, cbar_on=False, p_05_to_q=0.5)
	make_heatmap(None, ['All', 'AA', 'White'], "all sPTB vs TB", "./figurePanels/2A_xticks", 
		df=res, y_ticks=False, x_ticks=True, cbar_on=False, p_05_to_q=0.5)



def figure_2B():

	metabs = get_metabs(min_pres = 0, min_impute = False)
	md_metabs = get_md()
	data = metabs.join(md_metabs)
	data['dummy_var'] = 0

	fig, axs = plt.subplots(1, 3, figsize = (5, 3.5), sharey=True)
	sns.boxplot(data=data, x = 'dummy_var', y = 'diethanolamine', hue='PTB' , palette=ptb_colors, showfliers=False, ax = axs[0])
	sns.swarmplot(data=data, x = 'dummy_var', y = 'diethanolamine', hue='PTB', palette=['black', 'black'], dodge=True, size = 2.2, ax = axs[0])
	axs[0].set_xticks([])
	axs[0].set(ylim=(-6, 6))
	axs[0].get_legend().remove()

	sns.boxplot(data=data, x = 'dummy_var', y = 'choline', hue='PTB',  palette=ptb_colors, showfliers=False, ax = axs[1])
	sns.swarmplot(data=data, x = 'dummy_var', y = 'choline', hue='PTB', palette=['black', 'black'], dodge=True, size = 2.2, ax = axs[1])
	axs[1].set_xticks([])
	axs[1].get_legend().remove()

	sns.boxplot(data=data, x = 'dummy_var', y = 'betaine', hue='PTB',  palette=ptb_colors, showfliers=False, ax = axs[2])
	sns.swarmplot(data=data, x = 'dummy_var', y = 'betaine', hue='PTB', palette=['black', 'black'], dodge=True, size = 2.2, ax = axs[2])
	axs[2].set_xticks([])
	axs[2].get_legend().remove()

	for ax in axs:
	    ax.set_xlabel('')
	    ax.set_ylabel('')
	axs[1].tick_params(left=False)
	axs[2].tick_params(left=False)

	fig.subplots_adjust(wspace=0.05, hspace=0)
	axs[0].set_yticks([-4, -2, 0, 2, 4])

	plt.savefig(f'./figurePanels/2B.png', dpi = 800, bbox_inches = 'tight')
	plt.savefig(f'./figurePanels/2B.pdf', dpi = 800, bbox_inches = 'tight')
	plt.close()



def figure_2D():

	metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True, named_only=True)
	md = get_md().loc[metabs.index]
	metabs = metabs.loc[md.index]
	all_race_conds = list(zip((pd.Series(True, index = md.index), md.race == 1, md.race == 0), ('All', 'AA', 'White')))
	black_cond = [all_race_conds[1]]

	extremely_ptb_vs_rest = run_DA(metabs, md, ['extreme_sPTB'], ['very_sPTB', 'moderate_sPTB', 'TB'], False, black_cond)
	extremely_ptb_vs_rest['Cond'] = 'Below 28 (AA) vs 28 and above (AA)'
	very_and_extreme_ptb_vs_rest = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB'], ['moderate_sPTB', 'TB'], False, black_cond)
	very_and_extreme_ptb_vs_rest['Cond'] = 'Below 32 (AA) vs 32 and above (AA)'
	merged = pd.concat([very_and_extreme_ptb_vs_rest, extremely_ptb_vs_rest])
	merged['heatmap_q'] = multipletests(merged['p'], method='fdr_bh')[1]
	sig_hits = merged[merged['heatmap_q'] < 0.1]
	sig_hits.to_csv("differential_abundance_sig_hits/2D_sig_hits.csv")
	make_heatmap(None, list(merged.Cond.unique()), "Above GA thresh vs below GA thresh", "./figurePanels/2D", 
		df=merged, y_ticks=False, x_ticks=False, title_on=False, cbar_on=False, p_05_to_q=0.3)
	make_heatmap(None, list(merged.Cond.unique()), "Above GA thresh vs below GA thresh", "./figurePanels/2D_xticks", 
		df=merged, y_ticks=False, x_ticks=True, title_on=False, cbar_on=False, p_05_to_q=0.3)
