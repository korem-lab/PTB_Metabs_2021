import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from get_dfs import get_metabs, get_md
from definitions import ptb_colors

def figure_2B():

	sns.reset_orig()
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