import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from utils.get_dfs import get_metabs, get_md, get_all_nmpcs, get_v2_cts



def figure_S6ABC():

	nmpcs = get_all_nmpcs()
	metabs = get_metabs(min_impute=False)

	x = get_md()

	## Add nmpcs - need to add epsilon to histamine and tyrmaine nmpcs
	putrescine_nmpc = np.log10(nmpcs[['Putrescine NMPC']])
	x = x.join(putrescine_nmpc)

	histamine_nmpc = nmpcs[['Histamine NMPC']]
	eps = histamine_nmpc[histamine_nmpc > 0].min(axis=1).sort_values().iloc[0] / 10
	histamine_nmpc= np.log10(histamine_nmpc + eps)
	x = x.join(histamine_nmpc)

	tyramine_nmpc = nmpcs[['Tyramine NMPC']]
	eps = tyramine_nmpc[tyramine_nmpc > 0].min(axis=1).sort_values().iloc[0] / 10
	tyramine_nmpc= np.log10(tyramine_nmpc + eps)
	x = x.join(tyramine_nmpc)

	x = x.join(metabs[['putrescine', 'histamine', 'tyramine']])

	## S6A - putrescine
	fig = plt.figure(figsize=(4,4))
	sns.scatterplot(data=x, x='putrescine', y='Putrescine NMPC', s=55)
	plt.xlabel("")
	plt.ylabel("")
	rho, p = spearmanr(x['putrescine'], x['Putrescine NMPC'], nan_policy='omit')
	plt.yticks([-1, 0, 1, 2])
	plt.savefig("figurePanels/S6A.pdf", dpi=800)
	print(f"Agreement between\nmeasured and predicted putrescine,\nSpearman rho={rho}")

	## S6B - histamine
	fig = plt.figure(figsize=(4,4))
	sns.scatterplot(data=x, x='histamine', y='Histamine NMPC', s=55)
	plt.xlabel("")
	plt.ylabel("")
	rho, p = spearmanr(x['histamine'], x['Histamine NMPC'], nan_policy='omit')
	plt.yticks([-1, 0, 1, 2])
	plt.savefig("figurePanels/S6B.pdf", dpi=800)
	print(f"Agreement between\nmeasured and predicted histamine,\nSpearman rho={rho}")

	## S6C - tyramine
	fig = plt.figure(figsize=(4,4))
	sns.scatterplot(data=x, x='tyramine', y='Tyramine NMPC', s=55)
	plt.xlabel("")
	plt.ylabel("")
	rho, p = spearmanr(x['tyramine'], x['Tyramine NMPC'], nan_policy='omit')
	plt.yticks([-1, 0, 1, 2])
	plt.savefig("figurePanels/S6C.pdf", dpi=800)
	print(f"Agreement between\nmeasured and predicted tyramine,\nSpearman rho={rho}")

def figure_S6D():

	cts = get_v2_cts()
	rel_abs = cts.div(cts.sum(axis=1), axis=0)

	agora_matches = pd.read_excel("data/Supplementary Tables.xlsx", sheet_name='Table S9')
	agora_matches = agora_matches.dropna(subset=['AGORA2 Match'])

	rel_abs = rel_abs[agora_matches['Original Assignment']]
	cov = pd.DataFrame(rel_abs.sum(axis=1), columns=['Agora2_cov'])

	data = get_md()
	data = data[data.race != 2]
	data['race'] = data['race'].map({0:'white', 1:'black', 2:'other'})
	data['category'] = data['race'] + '_' + data['PTB']
	data['category'] = data['category'].map({'white_TB':'white_TB', 'white_sPTB':'white_sPTB', 
	                                            'black_TB':'Black women', 'black_sPTB':'Black women'})
	data = data.join(cov)
	plt.figure(figsize=(4,4))
	sns.boxplot(data=data, x='category', y='Agora2_cov', palette={'white_TB':'dodgerblue', 'white_sPTB':'firebrick', 'Black women':'darkgoldenrod'})
	sns.swarmplot(data=data, x='category', y='Agora2_cov', palette=['gray'], size=3)
	plt.xlabel("")
	plt.ylabel("")
	plt.xticks([], [])
	plt.savefig("figurePanels/S6D.pdf", dpi=800)



def figure_S6E():

	metabs = get_metabs(min_impute=False)
	true_NMPCs = get_all_nmpcs()

	res = []

	N_species = 95
	
	tym = metabs[['tyramine']].join(true_NMPCs['Tyramine NMPC']).dropna()
	putr = metabs[['putrescine']].join(true_NMPCs['Putrescine NMPC']).dropna()
	his = metabs[['histamine']].join(true_NMPCs['Histamine NMPC']).dropna()

	tym_rho = spearmanr(tym['tyramine'], tym['Tyramine NMPC'])[0]
	putr_rho = spearmanr(putr['putrescine'], putr['Putrescine NMPC'])[0]
	his_rho = spearmanr(his['histamine'], his['Histamine NMPC'])[0]

	res.append((N_species, tym_rho, putr_rho, his_rho))


	N_species = 85
	for i in range(1, 8):
	    curr_nmpcs = pd.read_csv(f"data/nmpcs/species_elim/standard_{i}.csv").T.set_index(0)

	    curr_nmpcs.index.name=None
	    curr_nmpcs.columns = curr_nmpcs.loc['NMPCs']
	    curr_nmpcs.columns.name=None
	    curr_nmpcs = curr_nmpcs.drop('NMPCs')
	    curr_nmpcs.index = [x[1:].replace('_', '.') for x in curr_nmpcs.index]
	    for c in curr_nmpcs.columns:
	        curr_nmpcs[c] = pd.to_numeric(curr_nmpcs[c])
	        
	    tym = metabs[['tyramine']].join(curr_nmpcs['EX_tym[fe]']).dropna()
	    putr = metabs[['putrescine']].join(curr_nmpcs['EX_ptrc[fe]']).dropna()
	    his = metabs[['histamine']].join(curr_nmpcs['EX_hista[fe]']).dropna()
	        
	    tym_rho = spearmanr(tym['tyramine'], tym['EX_tym[fe]'])[0]
	    putr_rho = spearmanr(putr['putrescine'], putr['EX_ptrc[fe]'])[0]
	    his_rho = spearmanr(his['histamine'], his['EX_hista[fe]'])[0]

	    res.append((N_species, tym_rho, putr_rho, his_rho))

	    N_species = N_species - 10
	    
	res = pd.DataFrame(res)
	res.columns = ['N_species', 'tyramine_rho', 'putrescine_rho', 'histamine_rho']
	
	fig, ax = plt.subplots(1,1, figsize=(6,4))
	ax.plot(res['N_species'], res['tyramine_rho'], marker='o', color='tab:blue', label='Tyramine')
	ax.plot(res['N_species'], res['putrescine_rho'], marker='o', color='tab:red', label='Putrescine')
	ax.plot(res['N_species'], res['histamine_rho'], marker='o', color='tab:green', label='Histmaine')
	ax.invert_xaxis()
	ax.set_xticks(res['N_species'])
	ax.set_ylim(-0.1, 1.1)
	ax.legend(bbox_to_anchor=(1.3, 1))
	plt.savefig("figurePanels/S6E.pdf", dpi=800, bbox_inches='tight')
