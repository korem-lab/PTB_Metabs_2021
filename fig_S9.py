import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from utils.definitions import cst_colors
from utils.get_dfs import get_md, get_sample_batches, get_metabs, get_metab_platforms, get_mcs
from utils.metabs_utils import metabolome_umap, do_barplot


def figures_S9AB():

	u, md = metabolome_umap()
	batches = get_sample_batches()

	fig = plt.figure(figsize = (4.5,4.5))
	kwargs = {'x': u[:,0], 'y':u[:,1]}
	hu = batches['LC MS/MS Pos Early, Pos Late, Polar Batch']
	sns.scatterplot(**kwargs, hue = hu, s=30, hue_order = sorted(hu.unique()), palette=['tab:blue', 'tab:orange'], ax = plt.subplot(111))
	plt.xticks([])
	_=plt.yticks([])
	plt.legend([],[], frameon=False)
	plt.savefig('./figurePanels/S9A.png', dpi = 800)
	plt.savefig('./figurePanels/S9A.pdf', dpi = 800)
	plt.close()

	fig = plt.figure(figsize = (4.5,4.5))
	kwargs = {'x': u[:,0], 'y':u[:,1]}
	hu = batches['LC MS/MS Neg Batch']
	sns.scatterplot(**kwargs, hue = hu, s=30, hue_order = sorted(hu.unique()), palette=['tab:red', 'tab:cyan', 'tab:brown'], ax = plt.subplot(111))
	plt.xticks([])
	_=plt.yticks([])
	plt.legend([],[], frameon=False)
	plt.savefig('./figurePanels/S9B.png', dpi = 800)
	plt.savefig('./figurePanels/S9B.pdf', dpi = 800)
	plt.close()


def figure_S9C():

	md = get_md()
	mcs = get_mcs()
	batches = get_sample_batches()
	x = md.join(mcs).join(batches)

	f = plt.figure(figsize=(4, 6))
	ax1 = f.add_subplot(211)
	do_barplot(x, 'Metabolomics Clusters', 'LC MS/MS Pos Early, Pos Late, Polar Batch', ['tab:blue', 'tab:orange'], None, 0, 0.35, ax=ax1, pvals=False, level=1, hue_order = np.sort(x['LC MS/MS Pos Early, Pos Late, Polar Batch'].unique()))
	ax2 = f.add_subplot(212, sharex=ax1)
	do_barplot(x, 'Metabolomics Clusters', 'LC MS/MS Neg Batch', ['tab:red', 'tab:cyan', 'tab:brown'], None, 0, 0.35, ax=ax2, pvals=False, level=1, hue_order = np.sort(x['LC MS/MS Neg Batch'].unique()))
	plt.subplots_adjust(hspace=0)
	plt.savefig(f"./figurePanels/S9C.png", dpi = 800, bbox_inches='tight')
	plt.savefig(f"./figurePanels/S9C.pdf", dpi = 800, bbox_inches='tight')


def figure_S9D():

	metabs = get_metabs(min_impute=False)
	md = get_md()
	md['PTB'] = md['PTB'].map({'sPTB':1, 'TB':0})

	sample_batches = get_sample_batches()
	sample_batches.columns = ['REST', 'NEG']

	m_platforms = get_metab_platforms()
	m_platforms['Extraction Platform'] = m_platforms['Extraction Platform'].map({'LC/MS Neg':'NEG',
															'LC/MS Polar':'REST',
															'LC/MS Pos Early':'REST',
															'LC/MS Pos Late':'REST'})


	whole_cohort = [True]*md.shape[0]
	black = md.race == 1
	white = md.race == 0
	conds = [('white', white), ('black', black), ('all', whole_cohort)]

	curr_metabs = ['glycerophosphoserine*', 'choline', 'spermine', 
					'(R)-3-hydroxybutyrylcarnitine', 'tyramine', 'glutamate, gamma-methyl ester', 
					'tartarate', 'ethyl beta-glucopyranoside', 'X - 25656', 'diethanolamine']

	coefs = np.empty((3, 10))
	ps = np.empty((3, 10))

	for j, m in enumerate(curr_metabs):
		plat = m_platforms.loc[m][0]

		x = md[['PTB']].join(metabs[m]).join(sample_batches[plat])
		x.rename(columns={m:'metabolite'}, inplace=True)

		for i, cond in enumerate(conds):
			print(cond[0])
			x2 = x.loc[cond[1]]
			formula = f'PTB ~ metabolite + {plat}'
			log_reg = smf.logit(formula, data=x2).fit(method='bfgs', maxiter=1000)
			curr_coeffs = pd.DataFrame(log_reg.summary2().tables[1])
			curr_coeffs.rename({'metabolite':m}, axis=0, inplace=True)
			print(curr_coeffs)
			m_coeff = curr_coeffs.loc[m, 'Coef.']
			m_p = curr_coeffs.loc[m, 'P>|z|']


			coefs[i, j] = m_coeff
			ps[i, j] = m_p

	coefs = pd.DataFrame(coefs)
	coefs.index = ['White', 'Black', 'All']
	coefs.columns = curr_metabs
	for_heatmap = np.exp(coefs)

	fig = plt.figure(figsize = (18, 6))
	sns.heatmap(for_heatmap, cmap='coolwarm', linewidths=2, linecolor='gray', center=1)
	plt.xlabel("")
	plt.ylabel("")
	plt.xticks([], [])
	plt.yticks([], [])
	plt.savefig("figurePanels/S9D.pdf", dpi=800, bbox_inches='tight')
