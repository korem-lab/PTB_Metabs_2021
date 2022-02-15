import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_pickle
from utils.get_dfs import get_md, get_v2_cts, get_v2_cts_1k, get_v2_aa 
from utils.definitions import cst_colors, race_colors
from utils.metabs_utils import microbiome_umap


def figure_1A():

	##### Fig 1A
	##### Microbiome umap colored by original CST
	u, md = microbiome_umap()

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
