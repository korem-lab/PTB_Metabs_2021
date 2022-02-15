import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import random
random.seed(100)
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, cut_tree

from get_dfs import get_metabs, get_md
from definitions import metabolite_annotations_path, mcs_path, mc_colors


def figure_S2():

	def labels_to_colors(labels, color_dict, color_palette):
		label_options = list(set(labels))
		label_options.sort()
		colors = []
		if color_palette == 'use_color_dict':
			for i in range(0, len(label_options)):
				colors.append(color_dict[label_options[i]])
		else:
			colors = sns.color_palette(color_palette, len(label_options))

		col_colors_dict = {}
		for i in range(0, len(label_options)):
			col_colors_dict[label_options[i]] = colors[i]
		item_colors = [col_colors_dict[x] for x in labels]

		legend_patches = []
		for key, value in col_colors_dict.items():
			legend_patches.append(mpatches.Patch(color=value, label=key))

		return item_colors, legend_patches


	### Load metadata
	md = get_md()

	### Load min-imputed log normalized robust z score metabolomics dataframe
	metabolomics_imputed = get_metabs(rzscore=True, min_impute=True)
	df_for_visualizing = metabolomics_imputed.copy()

	metabolite_annotations = pd.read_csv(metabolite_annotations_path, index_col=0)
	metabolite_annotations = metabolite_annotations.replace(np.nan, 'Partially Characterized Molecules')
	metabolite_annotations = metabolite_annotations[metabolite_annotations.index.isin(df_for_visualizing.columns)]


	mcs = pd.read_csv(mcs_path, index_col=0).loc[md.index]
	md['cluster'] = mcs['Metabolomics Clusters']
	mst_colors, mst_patches = labels_to_colors(list(md['cluster']), mc_colors, "use_color_dict")
	metab_sp_colors, metab_sp_patches = labels_to_colors(list(metabolite_annotations['Super pathway']), None, 'pastel')


	##### Average columns belonging to same metabolite cluster
	metabolite_dm = pd.DataFrame(squareform(pdist(df_for_visualizing.transpose().to_numpy(), metric='canberra')))
	metabolite_linkage = linkage(squareform(metabolite_dm.to_numpy()), method='ward', optimal_ordering=True)

	tree_cut = cut_tree(metabolite_linkage, n_clusters=3)
	tree_cut = [x[0] for x in tree_cut]
	metab_cluster_colors, metab_cluster_patches = labels_to_colors(list(tree_cut), None, 'pastel')

	N_samples = df_for_visualizing.shape[0]
	N_metabolites = df_for_visualizing.shape[1]
	metab_cluster_averages = np.zeros((N_samples, max(tree_cut)+1))

	for i in range(0, N_metabolites):
		curr_metab = df_for_visualizing.iloc[:, i]
		curr_cluster = tree_cut[i]
		metab_cluster_averages[:, curr_cluster] = metab_cluster_averages[:, (curr_cluster)] + curr_metab

	N_clusters = max(tree_cut)
	for i in range(0, N_clusters):
		metab_cluster_averages[:, i] = metab_cluster_averages[:, i] / tree_cut.count(i+1)

	metab_cluster_averages = pd.DataFrame(metab_cluster_averages)
	metab_cluster_averages.index = df_for_visualizing.index
	metab_cluster_averages.columns = ['metabolite cluster 1', 'metabolite cluster 2', 'metabolite cluster 3']

	metabolite_annotations['metabolite_cluster'] = tree_cut 
	metabolite_annotations['metabolite_cluster'] = metabolite_annotations['metabolite_cluster']


	plt.figure(figsize = (4.5,4.5))

	df_for_visualizing.index.name = None
	df_for_visualizing.columns.name = None
	metab_cols = df_for_visualizing.columns
	df_for_visualizing = df_for_visualizing.join(metab_cluster_averages)
	df_for_visualizing['color'] = mst_colors
	df_for_visualizing = df_for_visualizing.join(md[['cluster']])

	### Sorting for better visualization
	mca = df_for_visualizing[df_for_visualizing['cluster'] == 'A']
	mcb = df_for_visualizing[df_for_visualizing['cluster'] == 'B']
	mcc = df_for_visualizing[df_for_visualizing['cluster'] == 'C']
	mcd = df_for_visualizing[df_for_visualizing['cluster'] == 'D']
	mce = df_for_visualizing[df_for_visualizing['cluster'] == 'E']
	mcf = df_for_visualizing[df_for_visualizing['cluster'] == 'F']

	df_for_visualizing = pd.concat([mca, mcb, mcc, mcd, mce, mcf])
	mst_colors = list(df_for_visualizing['color'])

	g = sns.clustermap(df_for_visualizing[metab_cols], row_cluster=False, row_colors=[mst_colors], col_linkage=metabolite_linkage, col_colors=[metab_sp_colors], 
		xticklabels=False, yticklabels=False, cmap="vlag", colors_ratio=0.02, dendrogram_ratio=0.15, vmin=-4, vmax=4)

	metabolites_sp_legend = plt.legend(handles=metab_sp_patches, title='Metabolite Annotation', bbox_to_anchor=(1, .5), 
		bbox_transform=plt.gcf().transFigure, loc='upper left')
	plt.gca().add_artist(metabolites_sp_legend)

	plt.ylabel("")
	plt.xlabel("")
	plt.savefig(f"./figurePanels/S2.png", dpi=400, bbox_inches='tight')
	plt.savefig(f"./figurePanels/S2.pdf", dpi=400, bbox_inches='tight')
	plt.close()
