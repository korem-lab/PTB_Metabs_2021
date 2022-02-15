import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from definitions import NMPCs_path, ptb_colors
from get_dfs import get_md, get_metabs, get_tym_nmpcs
from metabs_utils import fisher_r_to_z

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
    tym_NMPC = get_tym_nmpcs(metabs=True)

    fig, axs = plt.subplots(1, 1, figsize = (2, 3.5), sharey=False)
    data = md.join(tym_NMPC, how='inner')
    data = data.loc[get_metabs().index, :]

    sns.boxplot(data=data[data.race != 2], x = 'race', y = 'TyramineNMPC', hue='PTB', palette=ptb_colors, showfliers=False, ax = axs, medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
    sns.swarmplot(data=data[data.race != 2], x = 'race', y = 'TyramineNMPC', hue='PTB', palette=['gray', 'gray'] , size = 2.2, ax = axs, dodge=True, alpha=1)
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
    tym_NMPC = get_tym_nmpcs(metabs=True)
    data = md.join(tym_NMPC, how='inner')
    data = data.join(metabs[['tyramine']].dropna(), how='inner')
    data = data[data.race != 2]
    data['race'] = data['race'].map({0:'white', 1:'black', 2:'other'})
    data['category'] = data['race'] + '_' + data['PTB']
    data['category'] = data['category'].map({'white_TB':'white_TB', 'white_sPTB':'white_sPTB', 
                                            'black_TB':'non-white', 'black_sPTB':'non-white',
                                            'other_TB':'non-white', 'other_sPTB':'non-white'})
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x='tyramine', y='TyramineNMPC', data=data, hue='category', legend=False, alpha=1, palette={'white_TB':'dodgerblue', 'white_sPTB':'firebrick', 'non-white':'darkgoldenrod'})
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([-2, -1, 0, 1, 2], fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f'./figurePanels/3D.png', dpi = 1200, bbox_inches = 'tight')
    plt.savefig(f'./figurePanels/3D.pdf', dpi = 1200, bbox_inches = 'tight')
    plt.close()