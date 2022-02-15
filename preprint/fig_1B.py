import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pandas import read_pickle
from get_dfs import get_metabs, get_md
from definitions import cst_colors


def figure_1B():
	##### Fig 1B
	##### Metabolome umap colored by original CST

	metabs = get_metabs(rzscore=True, min_impute=True)
	md = get_md()

	u = umap.UMAP(
	    n_neighbors=15,
	    min_dist=0.25,
	    n_components=2,
	    metric='canberra',
	    random_state=42).fit_transform(metabs)

	plt.close('all')
	fig = plt.figure(figsize = (4.5,4.5))
	kwargs = {'x': u[:,0], 'y':u[:,1]}
	hu = md.OrigCST
	sns.scatterplot(**kwargs, hue = hu, s=30, palette = cst_colors, hue_order = sorted(hu.unique()), ax = plt.subplot(111))

	plt.xticks([])
	_=plt.yticks([])
	plt.legend([],[], frameon=False)
	plt.savefig('./figurePanels/1B.png', dpi = 800)
	plt.savefig('./figurePanels/1B.pdf', dpi = 800)
	plt.close()