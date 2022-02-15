import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pandas import read_pickle
from get_dfs import get_metabs, get_md
from definitions import mc_colors, mcs_path


def figure_1C():

	##### Fig 1C
	##### Metabolome umap colored by Metabolome Clusters
	metabs = get_metabs(rzscore=True, min_impute=True)
	md = get_md()
	mcs = pd.read_csv(mcs_path, index_col=0).loc[metabs.index]

	u = umap.UMAP(
	    n_neighbors=15,
	    min_dist=0.25,
	    n_components=2,
	    metric='canberra',
	    random_state=42).fit_transform(metabs)

	plt.close('all')
	fig = plt.figure(figsize = (4.5,4.5))
	kwargs = {'x': u[:,0], 'y':u[:,1]}
	hu = mcs['Metabolomics Clusters']
	sns.scatterplot(**kwargs, hue = hu, s=30, palette = mc_colors, hue_order = sorted(hu.unique()), ax = plt.subplot(111))

	plt.xticks([])
	_=plt.yticks([])
	plt.legend([],[], frameon=False)
	plt.savefig('./figurePanels/1C.png', dpi = 800)
	plt.savefig('./figurePanels/1C.pdf', dpi = 800)
	plt.close()