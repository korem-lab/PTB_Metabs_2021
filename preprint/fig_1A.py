import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pandas import read_pickle
from get_dfs import get_md, get_v2_cts, get_v2_cts_1k, get_v2_aa 
from definitions import cst_colors


def figure_1A():

	##### Fig 1A
	##### Microbiome umap colored by original CST
	md = get_md(metabs=False)

	ct1k = get_v2_cts_1k(metabs=False)
	ct1k_lra = ct1k.replace(0, 0.0005).applymap(np.log10)
	md = md.loc[ct1k_lra.index]

	u = umap.UMAP(
	    n_neighbors=15,
	    min_dist=0.1,
	    n_components=2,
	    metric='braycurtis',
	    random_state=42).fit_transform(ct1k_lra)

	plt.close('all')
	fig = plt.figure(figsize = (4.5,4.5))
	kwargs = {'x': u[:,0], 'y':u[:,1]}

	hu = md.OrigCST
	sns.scatterplot(**kwargs, hue = hu.rename('CST'), s=30, palette = cst_colors, hue_order = sorted(hu.unique()), ax = plt.subplot(111))

	plt.xticks([])
	_=plt.yticks([])
	plt.legend([],[], frameon=False)
	plt.savefig('./figurePanels/1A.png', dpi = 800)
	plt.savefig('./figurePanels/1A.pdf', dpi = 800)
	plt.close()