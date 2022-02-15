import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_pickle
from get_dfs import get_metabs, get_md, get_v2_cts, get_v2_cts_1k, get_v2_aa
from definitions import cst_colors, mc_colors, race_colors, ptb_colors, mcs_path
from metabs_utils import add_p_val, large_fisher, do_barplot

def figures_S4ABCD():

	md_metabs = get_md()
	mcs = pd.read_csv(mcs_path, index_col=0).loc[md_metabs.index]
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
	plt.savefig(f"./figurePanels/S4A.png", dpi = 800, bbox_inches='tight')
	plt.savefig(f"./figurePanels/S4A.pdf", dpi = 800, bbox_inches='tight')

	#### stratified version of 1D CST enrichments in MCs
	f = plt.figure(figsize=(4, 6))
	ax1 = f.add_subplot(211)
	do_barplot(x[x.race == 0], 'Metabolomics Clusters', 'OrigCST', cst_colors, None, 0, 0.35, ax=ax1, hue_order = np.sort(x.OrigCST.unique()))
	ax2 = f.add_subplot(212, sharex=ax1)
	do_barplot(x[x.race == 1], 'Metabolomics Clusters', 'OrigCST', cst_colors, None, 0, 0.35, ax=ax2, hue_order = np.sort(x.OrigCST.unique()))
	plt.subplots_adjust(hspace=0)
	plt.savefig(f"./figurePanels/S4B.png", dpi = 800, bbox_inches='tight')
	plt.savefig(f"./figurePanels/S4B.pdf", dpi = 800, bbox_inches='tight')


	md_all = get_md(metabs=False)
	md_all = md_all.dropna(subset=['OrigCST'])
	x2 = md_all.sort_values(by = 'OrigCST')
	
	f = plt.figure(figsize=(4, 3))
	ax1 = f.add_subplot(111)
	do_barplot(x2, 'OrigCST', 'PTB', ptb_colors, None, 0, 0.35, ax=ax1)
	plt.savefig(f"./figurePanels/S4C.png", dpi = 800, bbox_inches='tight')
	plt.savefig(f"./figurePanels/S4C.pdf", dpi = 800, bbox_inches='tight')

	## Supp - versions of 1F, and 1G that are not race stratified
	f = plt.figure(figsize=(4, 3))
	ax1 = f.add_subplot(111)
	do_barplot(x, 'Metabolomics Clusters', 'PTB', ptb_colors, None, 0, 0.35, ax=ax1)
	plt.savefig(f"./figurePanels/S4D.png", dpi = 800, bbox_inches='tight')
	plt.savefig(f"./figurePanels/S4D.pdf", dpi = 800, bbox_inches='tight')