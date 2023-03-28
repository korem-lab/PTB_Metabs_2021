import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pandas import read_pickle
from utils.get_dfs import get_metabs, get_md, get_v2_cts, get_v2_cts_1k, get_v2_aa, get_mcs
from utils.definitions import cst_colors, mc_colors, race_colors, ptb_colors, early_ptb_colors
from utils.metabs_utils import large_fisher, do_barplot, microbiome_umap, metabolome_umap


def ext_data_fig_4ABEFG():

	md_metabs = get_md()
	mcs = get_mcs()
	md_metabs = md_metabs.join(mcs)
	x = md_metabs

	f = plt.figure(figsize=(4, 9))
	ax1 = f.add_subplot(311)
	do_barplot(x, 'Metabolomics Clusters', 'OrigCST', cst_colors, None, 0, 0.35, ax=ax1, pvals=False, level=0, hue_order = np.sort(x.OrigCST.unique()))
	ax2 = f.add_subplot(312, sharex=ax1)
	do_barplot(x[x.race == 0], 'Metabolomics Clusters', 'OrigCST', cst_colors, None, 0, 0.35, ax=ax2, pvals=False, level=0, hue_order = np.sort(x.OrigCST.unique()))
	ax3 = f.add_subplot(313, sharex=ax2)
	do_barplot(x[x.race == 1], 'Metabolomics Clusters', 'OrigCST', cst_colors, None, 0, 0.35, ax=ax3, pvals=False, level=0, hue_order = np.sort(x.OrigCST.unique()))
	plt.subplots_adjust(hspace=0)
	plt.savefig(f"./figurePanels/ext_data_4A.png", dpi = 800, bbox_inches='tight')
	plt.savefig(f"./figurePanels/ext_data_4A.pdf", dpi = 800, bbox_inches='tight')

	#### stratified version of 1D CST enrichments in MCs
	f = plt.figure(figsize=(4, 6))
	ax1 = f.add_subplot(211)
	do_barplot(x[x.race == 0], 'Metabolomics Clusters', 'OrigCST', cst_colors, None, 0, 0.35, ax=ax1, pvals=False, hue_order = np.sort(x.OrigCST.unique()))
	ax2 = f.add_subplot(212, sharex=ax1)
	do_barplot(x[x.race == 1], 'Metabolomics Clusters', 'OrigCST', cst_colors, None, 0, 0.35, ax=ax2, pvals=False, hue_order = np.sort(x.OrigCST.unique()))
	plt.subplots_adjust(hspace=0)
	plt.savefig(f"./figurePanels/ext_data_4B.png", dpi = 800, bbox_inches='tight')
	plt.savefig(f"./figurePanels/ext_data_4B.pdf", dpi = 800, bbox_inches='tight')


	md_all = get_md(metabs=False)
	md_all = md_all.dropna(subset=['OrigCST'])
	x2 = md_all.sort_values(by = 'OrigCST')
	
	f = plt.figure(figsize=(4, 3))
	ax1 = f.add_subplot(111)
	do_barplot(x2, 'OrigCST', 'PTB', ptb_colors, None, 0, 0.35, ax=ax1)
	plt.savefig(f"./figurePanels/ext_data_4E.png", dpi = 800, bbox_inches='tight')
	plt.savefig(f"./figurePanels/ext_data_4E.pdf", dpi = 800, bbox_inches='tight')

	## Supp - versions of 1F, and 1G that are not race stratified
	f = plt.figure(figsize=(4, 3))
	ax1 = f.add_subplot(111)
	do_barplot(x, 'Metabolomics Clusters', 'PTB', ptb_colors, None, 0, 0.35, ax=ax1)
	plt.savefig(f"./figurePanels/ext_data_4F.png", dpi = 800, bbox_inches='tight')
	plt.savefig(f"./figurePanels/ext_data_4F.pdf", dpi = 800, bbox_inches='tight')


	## early PTB enrichments in MCs
	x3 = x[x.race.isin([1])].copy()
	x3['early_sptb'] = x3['delGA'] < 32

	f = plt.figure(figsize=(4, 3))
	ax1 = f.add_subplot(111)
	do_barplot(x3, 'Metabolomics Clusters', 'early_sptb', early_ptb_colors, None, 0, 0.35, ax=ax1)
	plt.savefig(f"./figurePanels/ext_data_4G.pdf", dpi = 800, bbox_inches='tight')


def ext_data_fig_4CD():

	u, md = microbiome_umap()
	u = pd.DataFrame(u, index=md.index)
	md = md[md.race.isin([0, 1])]
	u = u.loc[md.index]

	plt.close('all')
	fig = plt.figure(figsize = (4.5,4.5))
	kwargs = {'x': u[0], 'y':u[1]}
	hu = md.race
	sns.scatterplot(**kwargs, hue = hu, s=30, palette = race_colors, hue_order = sorted(hu.unique()), ax = plt.subplot(111))
	plt.xticks([])
	_=plt.yticks([])
	plt.legend([],[], frameon=False)
	plt.xlabel("")
	plt.ylabel("")
	plt.savefig('./figurePanels/ext_data_4C.png', dpi = 800)
	plt.savefig('./figurePanels/ext_data_4C.pdf', dpi = 800)


	u, md = metabolome_umap()
	u = pd.DataFrame(u, index=md.index)
	md = md[md.race.isin([0, 1])]
	u = u.loc[md.index]

	plt.close('all')
	fig = plt.figure(figsize = (4.5,4.5))
	kwargs = {'x': u[0], 'y':u[1]}
	hu = md.race
	sns.scatterplot(**kwargs, hue = hu, s=30, palette = race_colors, hue_order = sorted(hu.unique()), ax = plt.subplot(111))
	plt.xticks([])
	_=plt.yticks([])
	plt.legend([],[], frameon=False)
	plt.xlabel("")
	plt.ylabel("")
	plt.savefig('./figurePanels/ext_data_4D.png', dpi = 800)
	plt.savefig('./figurePanels/ext_data_4D.pdf', dpi = 800)
