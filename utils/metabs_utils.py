import pandas as pd
import numpy as np
import glob
from scipy.stats import tiecorrect, rankdata, norm, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
from utils.get_dfs import get_md, get_v2_cts_1k, get_metabs
from pandas import Series, read_pickle, DataFrame, concat
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from pandas import get_dummies
from scipy.stats import fisher_exact


def wrpd_mnu(x, y, atleast):
    a = x.dropna()
    b = y.dropna()
    if len(a) < atleast or len(b) < atleast:
        return [np.nan] * 6
    if a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]:
        return [np.nan] * 6
    dms, dmp = mannwhitneyu(a,b, alternative='two-sided')
    if np.median(a) > np.median(b):
        dms = dms*-1
    return dms, dmp, np.median(a), np.median(b), len(a), len(b)


def wrpd_sprmn(atleast):
    def wrpd(x, y):
        m = x.notnull() & y.notnull()
        if m.sum() < atleast:
            return np.nan, np.nan, np.nan
        else:
            r, p = spearmanr(x[m].values, y[m].values)
            return (r, p, m.sum())
    return wrpd


def run_DA(df, md, case_labels, control_labels, shuffled, race_conds, seed=42, t_test=False, for_msea=False):
    
    GA_conditions = [
        (md['delGA'] < 28),
        (md['delGA'] >= 28) & (md['delGA'] < 32),
        (md['delGA'] >= 32) & (md['delGA'] < 37),
        (md['delGA'] > 37) ]
    GA_labels = ['extreme_sPTB', 'very_sPTB', 'moderate_sPTB', 'TB']
    md['GA_label'] = np.select(GA_conditions, GA_labels)


    res_all = []

    ### only include relevant cases and controls
    md = md[(md['GA_label'].isin(case_labels)) | md['GA_label'].isin(control_labels)]
    df = df.loc[md.index, :]

    md['case'] = md['GA_label'].isin(case_labels)

    for cond, cond_n in race_conds:

        cond_md = md[cond]
        cond_df = df[cond]

        if for_msea == False:
            threshold = max(cond_md.case.value_counts().min()/2, 10) ## Try this while keeping the threshold as 10, so we can get more p values
        else:
            threshold = 10

        if shuffled == True:
            cond_md.loc[:, 'case'] = cond_md['case'].sample(frac=1, random_state=seed).values ## validate that replacement is false

        res = DataFrame({m:wrpd_mnu(cond_df.loc[(cond_md.case == True), m], 
                    cond_df.loc[(cond_md.case == False), m], threshold) for m in cond_df.columns}, 
                    index = ['stat', 'p', 'sPTB_med', 'TB_med', 'sPTB_N', 'TB_N']).T

        res = res[res.p.notnull()]
        res['Cond'] = cond_n

        if res.shape[0] > 0:
            res_all.append(res)

    res_all = concat(res_all)
    res_all['overall_q'] = multipletests(res_all.p, method='fdr_bh')[1]
    res_all = res_all.sort_values(by='overall_q')
    return res_all


def run_correlations(aa, metabs, atleast):
    wrpd = wrpd_sprmn(atleast)
    res = pd.DataFrame({(a_col, m_col, wrpd(metabs[m_col], aa[a_col])) for m_col in metabs.columns for a_col in aa.columns})
    res.columns = ['Microbe', 'Metab', 'RP']
    res2 = res.join(res.RP.apply(pd.Series).rename(columns = {0:'R', 1:'p', 2:'N'})).drop('RP', axis = 1)
    res2 = res2[res2.p.notnull()]
    res2['q'] = multipletests(res2.p, method='fdr_bh')[1]
    return res2


def fisher_r_to_z(r1, r2, n1, n2):
    z1 = .5*np.log((1+r1)/(1-r1))
    z2 = .5*np.log((1+r2)/(1-r2))
    sezdiff = np.sqrt(1.06/(n1 - 3) + 1.06/(n2-3))
    ztest = (z1 - z2)/sezdiff
    alpha = 2*(1 - norm.cdf(abs(ztest),0,1))
    return alpha


def add_p_val(lft,rgt,y,h,p, t, ax):
    import matplotlib.pyplot as plt

    ax.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    if t:
        plt.text((lft + rgt) * .5, y, ("p<$10^{-5}$" if p<0.00001 else "p<$10^{-4}$" if p<0.0001 else "p<$10^{-3}$" if p<0.001 else f'{round(p, 3)}' if p<0.01 else f'{round(p, 3)}' if p<0.05 else 'n.s.'), ha='center', va='bottom', color='k')


def large_fisher(t):
    rpy2.robjects.numpy2ri.activate()
    stats = importr('stats')
    m = np.array(t)
    res = stats.fisher_test(m, workspace = 2e8)
    return res[0][0]


def do_barplot(x, x_col, y_col, palt, ofn, h = 1, pw=.2, ax = None, hue_order=None, pvals=True, level=1):

    a = x[[x_col, y_col]].value_counts().sort_index()
    a = (a.truediv(a.groupby(level = level).sum()) * 100).reset_index()
    a.columns = [x_col, y_col, 'Proportion (%)']
    
    if ax is None:
        fig = plt.figure(figsize = (4,2.7))

    sns.barplot(data= a, x = x_col, y='Proportion (%)', hue = y_col, hue_order = hue_order, palette = palt, ax = ax)

    for i, cst_ in enumerate(sorted(x[x_col].unique())):
        cont_tab = x.join(get_dummies(x[x_col]))[[cst_, y_col]].value_counts().unstack(0).fillna(0)

        if cont_tab.shape == (2,2):
            p = fisher_exact(cont_tab)[1]
        else:
            p = large_fisher(cont_tab)
        if p < 0.05 and pvals == True:
            add_p_val(i-pw, i+pw, a.loc[a[x_col] == cst_, 'Proportion (%)'].max() + 2, h, p, True, plt.gca())

    plt.ylim(0, a['Proportion (%)'].max() + 10)
    if ofn is not None or ax is not None:
        plt.legend([],[], frameon=False)
        plt.xticks([])
        plt.xlabel(None)
        plt.ylabel(None)
        if ax is None:
            plt.savefig(f"{ofn}.png", dpi = 800, bbox_inches='tight')
            plt.savefig(f"{ofn}.pdf", dpi = 800, bbox_inches='tight')



def msea(permuted_ps, real_DA_res, metabolite_annotations):

    ## remove internal commas from KEGG pathway names
    metabolite_annotations['Kegg pathway classes'] = [x.replace(", ", '_') if x == x else np.nan for x in
                                                      metabolite_annotations['Kegg pathway classes']]
    metabolite_annotations['Kegg pathways'] = [x.replace(", ", '_') if x == x else np.nan for x in
                                               metabolite_annotations['Kegg pathways']]

    res = []

    for set_type in metabolite_annotations.columns:

        curr_sets = list(metabolite_annotations[set_type].dropna().unique())

        ## get all of the individual kegg labels
        if set_type[0:4] == 'Kegg':
            curr_sets = [x.split(',') for x in curr_sets]
            temp = []
            for x in curr_sets:
                for y in x:
                    temp.append(y)
            curr_sets = set(temp)

        temp = metabolite_annotations[[set_type]].dropna()
        set_type_di = dict(zip(temp.index, temp[set_type]))

        for m_set in curr_sets:

            ## identify the compounds in the set
            compounds_in_set = []
            for k, v in set_type_di.items():
                if m_set in v:
                    compounds_in_set.append(k)

            in_set = real_DA_res.loc[real_DA_res.index.isin(compounds_in_set), :]
            out_of_set = real_DA_res.loc[~real_DA_res.index.isin(compounds_in_set), :]
            x = in_set['p']
            y = out_of_set['p']

            if len(x) >= 10:
                metab_set_raw_p = mannwhitneyu(x, y, alternative='two-sided')[1]
                shuffled_p_values = []
                for c in permuted_ps.columns:
                    df = permuted_ps[[c]]
                    in_set_shuffled = df.loc[df.index.isin(compounds_in_set), :]
                    out_of_set_shuffled = df.loc[~df.index.isin(compounds_in_set), :]
                    x_shuffled = in_set_shuffled[c].dropna()
                    y_shuffled = out_of_set_shuffled[c].dropna()
                    shuffled_p_values.append(mannwhitneyu(x_shuffled, y_shuffled, alternative='two-sided')[1])

                metab_set_p = max(len([x for x in shuffled_p_values if x < metab_set_raw_p]), 1) / len(
                    shuffled_p_values)
                res.append([set_type, m_set, len(x), metab_set_p])

    res = pd.DataFrame(res)
    res.columns = ['set_type', 'metabolite_set_name', 'DA_tests_in_set', 'p']
    return res


def microbiome_umap():

    md = get_md(metabs=False)

    ct1k = get_v2_cts_1k(metabs=False)
    ct1k_lra = ct1k.replace(0, 0.0005).applymap(np.log10)
    md = md.loc[ct1k_lra.index]

    u = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='braycurtis',
        random_state=47).fit_transform(ct1k_lra)

    # md_black_white = md[md.race.isin([0,1])]
    # ct1k_lra_black_white = ct1k_lra.loc[md_black_white.index]
    # from scipy.spatial.distance import pdist, squareform
    # from skbio.stats.distance import permanova, DistanceMatrix
    # dm = DistanceMatrix(squareform(pdist(ct1k_lra_black_white, metric='braycurtis')))
    # p_anova = permanova(dm, md_black_white['race'])
    # print(p_anova)

    return u, md


def metabolome_umap():

    metabs = get_metabs(rzscore=True, min_impute=True, named_only=False, drop_deprecated=True)
    md = get_md()

    u = umap.UMAP(
        n_neighbors=15,
        min_dist=0.25,
        n_components=2,
        metric='canberra',
        random_state=17).fit_transform(metabs)

    return u, md


def make_heatmap(ifn, conds, title, ofn, df=None, y_ticks=True, x_ticks=False, title_on=False, cbar_on=True, cbar_pos='left',  full=True, p_05_to_q=0.5):

    res_all = df
    res_sig = res_all[res_all.heatmap_q < 0.1]
    res_sig = res_sig[res_sig['Cond'].isin(conds)]
    res_all = res_all[res_all.index.get_level_values(0).isin(res_sig.index)]

    q_bins = [.01, .05, .1, p_05_to_q, 1.00001]
    res_all['q_discrete'] = res_all.heatmap_q.apply(lambda p: [b for b in q_bins if p < b][0])
    res_all['slogq'] = res_all.stat.apply(np.sign) * res_all.heatmap_q.apply(np.log10)

    conds = list(res_all.Cond.unique())
    x = res_all.set_index('Cond', append=True).slogq.unstack(1)[conds] ## new
    x = x.assign(m = x.mean(1)).sort_values('m').drop('m', axis = 1)
    x.columns = conds
    # sns.set(font="Calibri", font_scale = .9, style = 'whitegrid', context = 'talk')

    color_bins = ['#1d50a3', '#6a7bbb', '#abb1d8', '#e9eaf1', 'white', '#f3e8e2', '#f9896c', '#f76449', '#ff0013']
    cmap = ListedColormap(color_bins[1:-1])#.with_extremes(over = bins[-1], under = bins[1])
    cmap.set_over(color_bins[-1])
    cmap.set_under(color_bins[0])
    bounds = list(np.log10(q_bins[:-1])) + list(-np.log10(q_bins)[:-1][::-1])

    norm = BoundaryNorm(bounds, cmap.N)
    plt.close('all')
    x.index = x.index.to_series().str.capitalize().str.replace('*', '')

    N_metabs = len(set(res_all.index))
    N_conds = len(res_all['Cond'].unique())
    fig = plt.figure(figsize = (N_metabs, 2*N_conds))
    g = sns.heatmap(x[x.columns[::-1]].T, cmap=cmap, norm = norm, yticklabels = y_ticks, xticklabels = x_ticks, cbar=cbar_on, linewidths=2, linecolor='gray')

    ##remove colorbar tick labels
    cax = plt.gcf().axes[-1]
    cax.set_yticklabels([])

    _=g.set_xticklabels(g.get_xticklabels(), rotation=60, ha='right', fontsize = min(10, 10*(32/x[x.columns[::-1]].shape[0])), fontweight=10)
        
    if title_on == True:
        plt.title(title)

    height = x[x.columns[::-1]].T.shape[0]
    width = x[x.columns[::-1]].T.shape[1]
    g.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor='gray', lw=3))

    plt.savefig(f"{ofn}.png", dpi = 600, bbox_inches = 'tight')
    plt.savefig(f"{ofn}.pdf", dpi = 600, bbox_inches = 'tight')

    # sns.set_style("ticks")
    plt.close()


