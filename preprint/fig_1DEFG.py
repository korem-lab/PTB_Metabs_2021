import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_pickle
from get_dfs import get_md
from definitions import cst_colors, mc_colors, race_colors, ptb_colors, mcs_path
from metabs_utils import add_p_val, large_fisher, do_barplot


def figures_1DEFG():

    md_metabs = get_md()
    mcs = pd.read_csv(mcs_path, index_col=0).loc[md_metabs.index]
    md_metabs = md_metabs.join(mcs)
    x = md_metabs
    

    ### 1D and 1E MC enrichments for CSTs and race (not stratified)
    f = plt.figure(figsize=(4, 6))
    ax1 = f.add_subplot(211)
    do_barplot(x, 'Metabolomics Clusters', 'OrigCST', cst_colors, None, 0, 0.35, ax=ax1, hue_order = np.sort(x.OrigCST.unique()))
    ax2 = f.add_subplot(212, sharex=ax1)
    do_barplot(x[x.race != 2], 'Metabolomics Clusters', 'race', race_colors, None, 0, 0.35, level=1, ax=ax2, hue_order =np.sort(x[x.race != 2].race.unique()))
    plt.subplots_adjust(hspace=0)
    plt.savefig(f"./figurePanels/1DE.png", dpi = 800, bbox_inches='tight')
    plt.savefig(f"./figurePanels/1DE.pdf", dpi = 800, bbox_inches='tight')
    plt.close()


    #### 1F - PTB enrichments in CSTs
    md_all = get_md(metabs=False)
    x2 = md_all
    x2 = x2.sort_values(by = 'OrigCST')
    x2 = x2.dropna(subset=['OrigCST']) ## drop subjects without CST assignments
    f = plt.figure(figsize=(4, 6))
    ax1 = f.add_subplot(211)
    do_barplot(x2[x2.race == 0], 'OrigCST', 'PTB', ptb_colors, None, 0, 0.35, ax=ax1, hue_order = ['TB', 'sPTB'])
    ax1.set_xlim((-0.5, 5.5))
    ax2 = f.add_subplot(212, sharex=ax1)
    do_barplot(x2[x2.race == 1], 'OrigCST', 'PTB', ptb_colors, None, 0, 0.35, ax=ax2)
    ax2.set_xlim((-0.5, 5.5))
    ax2.set_yticks([0, 10, 20, 30])
    plt.subplots_adjust(hspace=0)
    plt.savefig(f"./figurePanels/1F.png", dpi = 800, bbox_inches='tight')
    plt.savefig(f"./figurePanels/1F.pdf", dpi = 800, bbox_inches='tight')
    plt.close()


    #### 1G - PTB enrichments in MCs
    ### Comments below are specific to the current MC clustering
    f = plt.figure(figsize=(4, 6))
    ax1 = f.add_subplot(211)
    do_barplot(x[x.race == 0], 'Metabolomics Clusters', 'PTB', ptb_colors, None, 0, 0.35, ax=ax1)
    ax1.set_xlim((-0.5, 5.5))
    ax2 = f.add_subplot(212, sharex=ax1)
    do_barplot(x[x.race == 1], 'Metabolomics Clusters', 'PTB', ptb_colors, None, 0, 0.35, ax=ax2)
    ax2.set_xlim((-0.5, 5.5))
    ax2.set_ylim((0, 39))
    ax2.set_yticks([0, 10, 20, 30])
    plt.subplots_adjust(hspace=0)
    plt.savefig(f"./figurePanels/1G.png", dpi = 800, bbox_inches='tight')
    plt.savefig(f"./figurePanels/1G.pdf", dpi = 800, bbox_inches='tight')
    plt.close()
