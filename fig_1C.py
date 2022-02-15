import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pandas import read_pickle
from utils.get_dfs import get_metabs, get_md, get_mcs
from utils.definitions import mc_colors
from utils.metabs_utils import metabolome_umap

def figure_1C():

	##### Fig 1C
	##### Metabolome umap colored by Metabolome Clusters
	u, md = metabolome_umap()

	mcs = get_mcs()
	hu = mcs['Metabolomics Clusters']

	plt.close('all')
	fig = plt.figure(figsize = (4.5,4.5))
	kwargs = {'x': u[:,0], 'y':u[:,1]}

	sns.scatterplot(**kwargs, hue = hu, s=30, palette = mc_colors, hue_order = sorted(hu.unique()), ax = plt.subplot(111))

	plt.xticks([])
	_=plt.yticks([])
	plt.legend([],[], frameon=False)
	plt.savefig('./figurePanels/1C.png', dpi = 800)
	plt.savefig('./figurePanels/1C.pdf', dpi = 800)
	plt.close()

