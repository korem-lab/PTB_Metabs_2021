import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.get_dfs import get_metabs, get_metab_annotations


def figure_S1():

	metabolites = get_metab_annotations()
	metabolites = metabolites.dropna()
	metabolomics = get_metabs(min_pres = 0, min_impute = False)

	plt.figure(figsize=(5,4))
	plt.rcParams['xtick.major.size'] = 5
	sns.set_style("ticks")
	sns.barplot(y=metabolites['Super pathway'].value_counts().index, x=metabolites['Super pathway'].value_counts(), color="gray")
	plt.title("")
	plt.xlabel("Number of metabolites")
	plt.tight_layout()
	plt.savefig(f"./figurePanels/S1A.png", dpi = 600, bbox_inches = 'tight')
	plt.savefig(f"./figurePanels/S1A.pdf", dpi = 600, bbox_inches = 'tight')

	plt.figure(figsize=(8,4))
	named_metabolites = [x for x in list(metabolomics.columns) if x[0:3] != 'X -']
	unnamed_metabolites = [x for x in list(metabolomics.columns) if x[0:3] == 'X -']

	y = list(metabolomics.shape[0] - metabolomics.isna().sum().sort_values())

	plt.plot(range(0, len(y)), y, color="gray")
	plt.fill_between(range(0, len(y)), y, color="gray", alpha=0.1)
	plt.xlabel("Number of metabolites")
	plt.ylabel("Number of samples")
	Eighty_thresh = round(metabolomics.shape[0]*0.8)
	twenty_thresh = round(metabolomics.shape[0]*0.2)

	eighty_line_x = 0
	twenty_line_x = 0

	for i in range(0, len(y)):
	    if eighty_line_x == 0:
	        if y[i] < Eighty_thresh:
	            eighty_line_x = i

	    if twenty_line_x == 0:
	        if y[i] < twenty_thresh:
	            twenty_line_x = i

	plt.axvline(x=eighty_line_x, color="black", linestyle='--')
	plt.axvline(x=twenty_line_x, color="black", linestyle='--')
	plt.xlim(0, metabolomics.shape[1]*1.02)
	plt.ylim(0, metabolomics.shape[0]*1.1)
	plt.title("")
	plt.text(eighty_line_x + 3.5, 230, ">80%")
	plt.text(twenty_line_x + 3.5, 230, "<20%")

	y2 = list(metabolomics.shape[0] - metabolomics[named_metabolites].isna().sum().sort_values())
	plt.plot(range(0, len(y2)), y2, color="blue")
	plt.fill_between(range(0, len(y2)), y2, color="blue", alpha=0.1)


	from matplotlib.lines import Line2D
	custom_lines = [Line2D([0], [0], color="gray", lw=4),
	                Line2D([0], [0], color="blue", lw=4)]

	plt.legend(custom_lines, ['All metabolites', 'Named metabolites', 'Unnamed metabolites'], bbox_to_anchor=(1.05, 1), loc="upper left")
	plt.tight_layout()
	plt.savefig(f"./figurePanels/S1B.png", dpi = 600, bbox_inches = 'tight')
	plt.savefig(f"./figurePanels/S1B.pdf", dpi = 600, bbox_inches = 'tight')
