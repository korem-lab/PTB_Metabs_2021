import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import random
random.seed(100)
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
from skbio.stats.ordination import pcoa
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from itertools import permutations

from utils.get_dfs import get_metabs, get_md, get_metab_annotations, get_mcs
from utils.definitions import mc_colors


## MC clustering inertia and gap statistic
def ext_data_fig_3AB():

	def generate_random_reference_uniform(df):
	    rd_features=[]
	    for i in range(len(df.iloc[0,:])):
	        a = np.min(df.iloc[:,i])
	        b = np.max(df.iloc[:,i])
	        rd_features.append(np.random.uniform(a,b,len(df)))
	    rd_df = pd.DataFrame(data=np.array(rd_features).T, columns=df.columns, index=df.index)
	    return rd_df
	
	# specify parameters
	kmax = 15
	internal_inertia = True

	df = get_metabs(rzscore=True, min_impute=True, named_only=False)
	dist = pairwise_distances(df, metric="canberra")

	wcss = []
	gaps = []
	ch_index = []
	sil = []

	for k in range(1, kmax+1):
	    cluster = KMedoids(n_clusters=k, metric="precomputed", random_state=42).fit(dist)
	    curr_wcss = cluster.inertia_
	    wcss.append(curr_wcss)
	    
	    # generate random distributions and cluster in order to calculate gap stat
	    rd_wcss = []
	    for ref_num in range(10):
	        rd_curr_wcss = 0
	        rd_df = generate_random_reference_uniform(df)
	        rd_dist = pairwise_distances(rd_df, metric='canberra')
	        rd_cluster = KMedoids(n_clusters=k, metric="precomputed", random_state=42).fit(rd_dist)
	        rd_curr_wcss = rd_cluster.inertia_ # use the distance metric passed in cluster
	        rd_wcss.append(rd_curr_wcss)

	    ### calculate the gap statistic for current k
	    gap = np.log(np.mean(rd_wcss)) - np.log(curr_wcss)
	    gaps.append(gap)


	plt.figure(figsize=(8,4))
	plt.plot(list(range(1,kmax+1)), wcss, '-o')
	plt.xticks(list(range(1,kmax+1)))
	plt.savefig("figurePanels/ext_data_3A.png", dpi=800, bbox_inches='tight')
	plt.savefig("figurePanels/ext_data_3A.pdf", dpi=800, bbox_inches='tight')
	plt.figure(figsize=(8,4))
	plt.plot(list(range(1,kmax+1)), gaps, '-o')
	plt.xticks(list(range(1,kmax+1)))
	plt.savefig("figurePanels/ext_data_3B.png", dpi=800, bbox_inches='tight')
	plt.savefig("figurePanels/ext_data_3B.pdf", dpi=800, bbox_inches='tight')


## Heatmap of metabolite levels
def ext_data_fig_3C():

	def labels_to_colors(labels, color_dict, color_palette):

		label_options = list(set(labels))
		label_options.sort()
		colors = sns.color_palette(color_palette, len(label_options))

		col_colors_dict = {}
		for i in range(0, len(label_options)):
			col_colors_dict[label_options[i]] = colors[i]
		item_colors = [col_colors_dict[x] for x in labels]

		legend_patches = []
		for key, value in col_colors_dict.items():
			legend_patches.append(mpatches.Patch(color=value, label=key))

		return item_colors, legend_patches

	### Load min-imputed log normalized robust z score metabolomics dataframe
	metabs = get_metabs(rzscore=True, min_impute=True, named_only=False)
	mcs = get_mcs()

	metabolite_annotations = get_metab_annotations()
	metabolite_annotations = metabolite_annotations.replace(np.nan, 'Partially Characterized Molecules')
	metabolite_annotations = metabolite_annotations[metabolite_annotations.index.isin(metabs.columns)]
	metab_sp_colors, metab_sp_patches = labels_to_colors(list(metabolite_annotations['Super pathway']), None, 'pastel')

	metabolite_dm = pd.DataFrame(squareform(pdist(metabs.transpose().to_numpy(), metric='canberra')))
	metabolite_linkage = linkage(squareform(metabolite_dm.to_numpy()), method='ward', optimal_ordering=True)

	plt.figure(figsize = (4.5,4.5))

	metabs.index.name = None
	metabs.columns.name = None
	metab_cols = metabs.columns
	metabs = metabs.join(mcs)
	metabs = metabs.sort_values(by='Metabolomics Clusters')

	g = sns.clustermap(metabs[metab_cols], row_cluster=False, row_colors=[metabs['Metabolomics Clusters'].map(mc_colors)], col_linkage=metabolite_linkage, col_colors=[metab_sp_colors], 
		xticklabels=False, yticklabels=False, cmap="vlag", colors_ratio=0.02, dendrogram_ratio=0.15, vmin=-4, vmax=4)

	metabolites_sp_legend = plt.legend(handles=metab_sp_patches, title='Metabolite Annotation', bbox_to_anchor=(1, .5), 
		bbox_transform=plt.gcf().transFigure, loc='upper left')
	plt.gca().add_artist(metabolites_sp_legend)

	plt.ylabel("")
	plt.xlabel("")
	plt.savefig(f"./figurePanels/ext_data_3C.png", dpi=150, bbox_inches='tight')
	plt.savefig(f"./figurePanels/ext_data_3C.pdf", dpi=150, bbox_inches='tight')
	plt.close()


def ext_data_fig_3DEF():

	metabs = get_metabs(rzscore=True, min_impute=True, named_only=False)
	md = get_md()
	mcs = get_mcs()

	## S2D - pca
	pca = PCA(n_components=2)
	X = pca.fit_transform(metabs)
	fig = plt.figure(figsize = (4.5,4.5))
	kwargs = {'x': X[:,0], 'y':X[:,1]}
	hu = mcs['Metabolomics Clusters']
	sns.scatterplot(**kwargs, hue = hu, s=30, palette = mc_colors, hue_order = sorted(hu.unique()), ax = plt.subplot(111))
	plt.xticks([])
	_=plt.yticks([])
	plt.legend([],[], frameon=False)
	plt.savefig("./figurePanels/ext_data_3D.pdf", dpi=800)

	## S2E - pcoa
	dm = squareform(pdist(metabs, metric='canberra'))
	X = pcoa(dm, method='eigh', number_of_dimensions=2)
	fig = plt.figure(figsize = (4.5,4.5))
	kwargs = {'x': X.samples['PC1'].values, 'y':X.samples['PC2'].values}
	hu = mcs['Metabolomics Clusters']
	sns.scatterplot(**kwargs, hue = hu, s=30, palette = mc_colors, hue_order = sorted(hu.unique()), ax = plt.subplot(111))
	plt.xticks([])
	_=plt.yticks([])
	plt.legend([],[], frameon=False)
	plt.savefig("./figurePanels/ext_data_3E.pdf", dpi=800)

	## S2F - tSNE
	tsne = TSNE(metric='canberra', n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
	X = tsne.fit_transform(metabs)
	fig = plt.figure(figsize = (4.5,4.5))
	kwargs = {'x': X[:,0], 'y':X[:,1]}
	hu = mcs['Metabolomics Clusters']
	sns.scatterplot(**kwargs, hue = hu, s=30, palette = mc_colors, hue_order = sorted(hu.unique()), ax = plt.subplot(111))
	plt.xticks([])
	_=plt.yticks([])
	plt.legend([],[], frameon=False)
	plt.savefig("./figurePanels/ext_data_3F.pdf", dpi=800)


def ext_data_fig_3G():

	md = get_md()
	mcs = get_mcs()
	metabs = get_metabs(min_impute=True, rzscore=True, named_only=False, drop_deprecated=True)


	perms = list(permutations(range(0, 6)))
	perms = [list(x) for x in perms]
	res = []

	for i in range(0, 100):

		rs = metabs.sample(frac=0.9, random_state=i*100)

		curr_mcs = mcs.loc[rs.index]

		dists = pairwise_distances(rs, metric="canberra")
		cluster = KMedoids(n_clusters=6, metric="precomputed", random_state=42).fit(dists)
		curr_mcs['new_c'] = cluster.labels_
		cm = pd.crosstab(curr_mcs['Metabolomics Clusters'], curr_mcs['new_c'])

		max_matching = 0
		best_p = 0
		for p in perms:
			curr_cm = cm[p]
			N_matching = np.sum(np.diagonal(curr_cm.values))
			if N_matching > max_matching:
				max_matching = N_matching
				best_p = p

		same_c_frac = max_matching / rs.shape[0]
		res.append(same_c_frac)

	plt.figure(figsize=(8,5))
	sns.histplot(res)
	print(res)
	print(len(res))
	print(len([x for x in res if x > 0.95]))
	plt.title("")
	plt.xlabel("")
	plt.ylabel("")
	plt.axvline(np.mean(res), color='red')
	print(f"mean of distribution in extended data fig 3G: {np.mean(res)}")
	plt.savefig("./figurePanels/ext_data_3G.pdf", dpi=800, bbox_inches='tight')
