import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_pickle
from utils.get_dfs import get_metabs, get_md
from utils.definitions import cst_colors, race_colors
from utils.metabs_utils import metabolome_umap

def figure_1B():
	##### Fig 1B
	##### Metabolome umap colored by original CST

	u, md = metabolome_umap()

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
