import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.definitions import ptb_colors
from utils.get_dfs import get_md, get_metabs, get_tym_nmpcs, get_v2_aa
from utils.metabs_utils import fisher_r_to_z, run_correlations, run_DA


def figure_3A():

    metabs = get_metabs(min_pres = 0, min_impute = False, rzscore=True)
    md = get_md()
    all_race_conds = list(zip((pd.Series(True, index = md.index), md.race == 1, md.race == 0), ('All', 'AA', 'White')))
    black_cond = [all_race_conds[1]]

    DA_res = run_DA(metabs, md, ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB'], ['TB'], False, all_race_conds)
    DA_res = DA_res[DA_res['overall_q'] < 0.1]
    DA_metabs = list(set(DA_res.index))
    metabs = metabs[DA_metabs]

    aa = get_v2_aa(metabs=True)
    md = get_md(metabs=True)
    aa = aa.replace(0, np.nan)
    aa = aa.loc[:, aa.notnull().sum() > 19]
    metabs = metabs.loc[aa.index]

    corrs_all = run_correlations(aa, metabs, 50).set_index(['Microbe', 'Metab'])
    corrs_all['abs(R)'] = np.abs(corrs_all['R'])
    corrs_all = corrs_all[((corrs_all['abs(R)'] > 0.25) & (corrs_all['N'] > 50) & (corrs_all['q'] < 0.1))]

    corrs_all.to_csv("./microbe_metabolite_networks/3A_edges.tsv", sep='\t')



def figure_3B():

    ##### Fig 3B - tyramine metabolite measurement boxplots
    metabs = get_metabs(min_pres = 0, min_impute = False)
    md_metabs = get_md()
    data = metabs.join(md_metabs)

    fig, axs = plt.subplots(1, 1, figsize = (2, 3.5), sharey=False)
    data = metabs.join(md_metabs)
    sns.boxplot(data=data[data.race != 2], x = 'race', y = 'tyramine', hue='PTB', palette=ptb_colors, showfliers=False, ax = axs, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
    sns.swarmplot(data=data[data.race != 2], x = 'race', y = 'tyramine', hue='PTB', palette=['gray', 'gray'] , size = 2.2, ax = axs, dodge=True, alpha=1)
    axs.set_yticks([-2, -1, 0, 1, 2])
    axs.set_xlabel('')
    axs.set_ylabel('')
    axs.set_xticks([])
    axs.get_legend().remove()
    plt.ylim([-2.5, 2.6])
    plt.savefig(f'./figurePanels/3B.png', dpi = 800, bbox_inches = 'tight')
    plt.savefig(f'./figurePanels/3B.pdf', dpi = 800, bbox_inches = 'tight')
    plt.close()


def figure_3C():

    md = get_md(metabs=True)
    tym_NMPC = get_tym_nmpcs()

    fig, axs = plt.subplots(1, 1, figsize = (2, 3.5), sharey=False)
    data = md.join(tym_NMPC, how='inner')
    data = data.loc[get_metabs().index, :]

    sns.boxplot(data=data[data.race != 2], x = 'race', y = 'Tyramine NMPC', hue='PTB', palette=ptb_colors, showfliers=False, ax = axs, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
    sns.swarmplot(data=data[data.race != 2], x = 'race', y = 'Tyramine NMPC', hue='PTB', palette=['gray', 'gray'] , size = 2.2, ax = axs, dodge=True, alpha=1)
    axs.set_yticks([-1, 0, 1, 2])
    axs.set_xlabel('')
    axs.set_ylabel('')
    axs.set_xticks([])
    axs.get_legend().remove()
    plt.ylim([-2, 3])
    plt.savefig('./figurePanels/3C.png', dpi = 800, bbox_inches = 'tight')
    plt.savefig('./figurePanels/3C.pdf', dpi = 800, bbox_inches = 'tight')
    plt.close()


def figure_3D():

    md = get_md(metabs=True)
    metabs = get_metabs(min_impute = False)
    tym_NMPC = get_tym_nmpcs()
    data = md.join(tym_NMPC, how='inner')
    data = data.join(metabs[['tyramine']].dropna(), how='inner')
    data = data[data.race != 2]
    data['race'] = data['race'].map({0:'white', 1:'black', 2:'other'})
    data['category'] = data['race'] + '_' + data['PTB']
    data['category'] = data['category'].map({'white_TB':'white_TB', 'white_sPTB':'white_sPTB', 
                                            'black_TB':'non-white', 'black_sPTB':'non-white',
                                            'other_TB':'non-white', 'other_sPTB':'non-white'})
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x='tyramine', y='Tyramine NMPC', data=data, hue='category', legend=False, alpha=1, palette={'white_TB':'dodgerblue', 'white_sPTB':'firebrick', 'non-white':'darkgoldenrod'})
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([-2, -1, 0, 1, 2], fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f'./figurePanels/3D.png', dpi = 1200, bbox_inches = 'tight')
    plt.savefig(f'./figurePanels/3D.pdf', dpi = 1200, bbox_inches = 'tight')
    plt.close()
