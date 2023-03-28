import pyreadr
import numpy as np
import pandas as pd
from pandas import Series
import os

from utils.definitions import elovitz_ct_path, supp_tables_path


def subsample_otu(o, thresh):
    if o.sum() < thresh:
        return np.nan
    v = o.values

    thresh = int(thresh)
    np.random.seed(42)
    a = np.random.permutation(np.arange(v.sum()))[:thresh]
    b = np.vectorize(lambda i: (a < i).sum())(v.cumsum())
    b[1:] -= b[:-1]
    return Series(b, index=o.index, name=o.name)


### rarefaction
def subsample_otudf(o, thresh):
    o2 = o.loc[o.sum(1).sort_values(ascending=False).index].apply(lambda ot: subsample_otu(ot, thresh), axis=1).truediv(
        thresh).dropna()

    o2 = o2.loc[o.index[o.index.isin(o2.index)]]
    o2 = o2.loc[:, o2.sum() > 0]
    return o2


def get_v2_cts(metabs=True):
    ct = pyreadr.read_r(elovitz_ct_path)['ct']
    ret = ct[ct.index.to_series().apply(lambda v: v.split('.')[1] == 'V2')]

    if metabs:
        m = get_metabs()
        ret = ret.loc[m.index]
    else:
        cts_subjects = ret.index
        md_subjects = get_md(metabs=False).index
        both = set(cts_subjects).intersection(md_subjects)
        both = list(both)
        both.sort()
        ret = ret.loc[both]

    return ret


def get_v2_cts_1k(metabs=True):
    ret = subsample_otudf(get_v2_cts(metabs), 1000)
    return ret


def get_v2_aa(metabs=True):
    cts = get_v2_cts(metabs)
    bl = pyreadr.read_r(elovitz_ct_path)['mt'][['bLoad']]

    y = cts.truediv(cts.sum(1), axis=0).join(bl, how='inner')
    y = y.iloc[:, :-1].multiply(y.bLoad, axis=0).dropna(how='all', axis=0)

    if metabs:
        m = get_metabs()
        keep = set(m.index).intersection(set(y.index))
        y = y.loc[keep]

    return y


def get_bl():
    cts = get_v2_cts()
    bl = pyreadr.read_r(elovitz_ct_path)['mt'][['bLoad']]
    return bl


def get_metabs(rzscore=True, min_pres=5, min_impute=True, vol_norm=True, compids=False,
               named_only=True, drop_deprecated=False):
    def _fmin(d):
        return d.fillna(d.min())

    def _rzsc(d, q = 0.05):
        return (d-d.median()) / (d[(d >= d.quantile(q)) & (d <= d.quantile(1-q))]).std()

    ret = pd.read_excel(supp_tables_path, sheet_name='Table S1', index_col=0)
    v = 70 / ret['Volume extracted (ul)']
    ret = ret.drop(['Volume extracted (ul)', 'LC MS/MS Pos Early, Pos Late, Polar Batch', 'LC MS/MS Neg Batch'], axis=1)

    # drop the two deprecated features and rename the one identified feature
    if drop_deprecated:
        if 'X - 25828' in ret.columns and 'X - 24697' in ret.columns:
            ret = ret.drop(['X - 25828', 'X - 24697'], axis=1)
    ret = ret.rename(columns={'X - 12849':'menthol glucuronide'})

    if vol_norm:
        ret = ret.mul(v, axis=0)

    if rzscore:
        ret = _rzsc(ret.apply(np.log10))

    if min_impute:
        ret = _fmin(ret)

    if min_pres > 0:
        # ret = ret.loc[:, (ret > ret.min()).sum() > min_pres] ## old method
        ret = ret.loc[:, (~ret.isna()).sum() >= min_pres]

    if compids:
        m_ids = pd.read_excel("data/Supplementary Tables.xlsx", sheet_name='Table S4')
        di = dict(zip(m_ids['Metabolite'], m_ids['COMP ID']))
        ret.columns = [str(di[x]) for x in ret.columns]

    if named_only:
        ret = ret[[x for x in ret.columns if x[0:3] != 'X -']]

    return ret


def get_cid_di():
    m_ids = pd.read_excel("data/Supplementary Tables.xlsx", sheet_name='Table S4')
    di = dict(zip(m_ids['COMP ID'].astype(str), m_ids['Metabolite']))
    return di


def get_md(metabs=True):

    md = pyreadr.read_r(elovitz_ct_path)['mt']

    if os.path.exists("data/age_bmi.csv"):
        age_bmi = pd.read_csv("data/age_bmi.csv", index_col=0)
        md = md.join(age_bmi)
        md['v2_bmi'] = pd.to_numeric(md['v2_bmi'].replace('.', np.nan))

    ## restrict to V2 and non-mPTB subjects
    md = md[md.visit == 'V2']
    md = md[md.mPTB != 1]

    ## swap cst IV-A and cst IV-B assignments
    md['cst'] = md['cst'].map({'I':'I',
                        'II':'II',
                        'III':'III',
                        'IV-A':'IV-B',
                        'IV-B':'IV-A',
                        'V':'V'})
    if metabs:
        m = get_metabs()
        md = md.loc[m.index]

    md['PTB'] = md.PTB.replace(0, 'TB').replace(1, 'sPTB')
    md['PTB'] = md['PTB'].astype(str)
    md.rename(columns={'cst':'OrigCST'}, inplace=True)

    ## correct race == 99 subject and 1515.V2
    md['race'] = md['race'].replace(99, 1)
    md.loc['1515.V2', 'race'] = 1

    return md


def get_mcs():
    ret = pd.read_excel(supp_tables_path, sheet_name='Table S2', index_col=0)
    return ret


def get_tym_nmpcs():
    nmpcs = pd.read_excel(supp_tables_path, sheet_name = 'Table S8', index_col=0)
    nmpcs = nmpcs[['Tyramine NMPC']]
    eps = nmpcs[nmpcs > 0].min(axis=1).sort_values().iloc[0] / 10
    nmpcs = np.log10(nmpcs + eps)
    return nmpcs


def get_metab_annotations():
    ret = pd.read_excel(supp_tables_path, sheet_name = 'Table S4', index_col=0)
    ret = ret.drop('Extraction Platform', axis=1)

    m_names = list(ret.index)
    for i, m in enumerate(m_names):
        if m[0:2] == 'X-':
            m_names[i] = m_names[i].replace("-", " - ")
    ret.index = m_names

    return ret


def get_all_nmpcs():
    nmpcs = pd.read_excel(supp_tables_path, sheet_name = 'Table S8', index_col=0)
    return nmpcs


def get_sample_batches():
    ret = pd.read_excel(supp_tables_path, sheet_name='Table S1', index_col=0)
    ret = ret[['LC MS/MS Pos Early, Pos Late, Polar Batch', 'LC MS/MS Neg Batch']]
    return ret


def get_metab_platforms():
    ret = pd.read_excel("data/Supplementary Tables.xlsx", sheet_name = 'Table S4', index_col=0)
    ret = ret[['Extraction Platform']]
    return ret

def get_unknowns_mass_RI():
    ret = pd.read_excel(supp_tables_path, sheet_name='Table S13')
    ret = ret[['Metabolite id', 'm/z', 'RI']]
    ret.rename(columns={'m/z':'MASS'}, inplace=True)
    ret = ret.set_index('Metabolite id')
    ret.index = [x.replace('-', ' - ') for x in ret.index]
    return ret


def get_vp():
    ret = pd.read_csv("data/vp.csv", index_col=0)
    return ret


def get_usearch_otus(): # note that OTU table is generated by Usearch, see Method section for more details
    otu = pd.read_csv("data/otu.csv").set_index("sample_id")
    return otu


def get_clin_df_for_pred():
    md = get_md()
    md = md[["maternal age", "v2_bmi", "race", "nullip", "hxPTB"]]
    md['hxPTB'] = md['hxPTB'].fillna(0)
    return md


def get_ghartey_2016():
    v2_black_2016_all = pd.read_excel("data/2016_v2.XLSX", sheet_name="OrigScale")
    tmp = pd.DataFrame()
    tmp['sample_id'] = v2_black_2016_all['Unnamed: 1'][8:]

    # get the comp id
    v2_black_2016_cid = pd.DataFrame(data=v2_black_2016_all.iloc[8:,[1,4]].values, columns=["metabolites","comp_id"])
    v2_black_2016_cid = v2_black_2016_cid.set_index(["metabolites"])
    v2_black_2016_cid = v2_black_2016_cid.to_dict()['comp_id']

    # Append the metabolomics data with the associated Biochemical
    v2_black_2016 = v2_black_2016_all.iloc[8:,13:]
    v2_black_2016
    tmp = pd.concat([tmp, v2_black_2016], axis=1)
    tmp = tmp.set_index("sample_id")
    tmp = tmp.T

    # save the metadata
    v2_black_2016_metadata = pd.DataFrame(index=tmp.index,
                                          columns=v2_black_2016_all.iloc[:8,12].values,
                                          data=v2_black_2016_all.iloc[:8,13:].T.values)
    v2_black_2016_metadata.index.name = "sample_id"

    v2_black_2016_metadata["ptb"] = 2
    for i in range(len(v2_black_2016_metadata.iloc[:,7])):
        if v2_black_2016_metadata.iloc[i,7] == "Preterm":
            v2_black_2016_metadata.iloc[i,8] = 1
        else:
            v2_black_2016_metadata.iloc[i,8] = 0

    v2_black_2016_metadata = v2_black_2016_metadata.sort_index()

    # apply log
    v2_black_2016 = np.log10(tmp.fillna(0))
    v2_black_2016 = v2_black_2016.replace(-np.inf, np.nan)

    # map the comp id
    v2_black_2016.columns = v2_black_2016.columns.to_series().map(v2_black_2016_cid)

    # save
    v2_black_2016.index.name = "sample_id"
    v2_black_2016 = v2_black_2016.loc[v2_black_2016_metadata["SAMPLE ID"].sort_values().index]
    v2_black_2016 = v2_black_2016.sort_index()

    v2_black_2016.columns = [str(c) for c in v2_black_2016.columns]

    return v2_black_2016, v2_black_2016_metadata


def get_ghartey_2014():
    v2_white_2014_all = pd.read_excel("data/2014_v2.XLSX", sheet_name="OrigScale")
    tmp = pd.DataFrame()
    tmp['sample_id'] = v2_white_2014_all['Unnamed: 1'][12:]

    # get the comp id
    v2_white_2014_cid = pd.DataFrame(data=v2_white_2014_all.iloc[12:,[1,4]].values, columns=["metabolites","comp_id"])
    v2_white_2014_cid = v2_white_2014_cid.set_index(["metabolites"])
    v2_white_2014_cid = v2_white_2014_cid.to_dict()['comp_id']

    # Append the metabolomics data with the associated Biochemical
    v2_white_2014 = v2_white_2014_all.iloc[12:,13:]
    tmp = pd.concat([tmp, v2_white_2014], axis=1)
    tmp = tmp.set_index("sample_id")
    tmp = tmp.T

    # save the metadata
    v2_white_2014_metadata = pd.DataFrame(index=tmp.index,
                                          columns=v2_white_2014_all.iloc[:12,12].values,
                                          data=v2_white_2014_all.iloc[:12,13:].T.values)
    v2_white_2014_metadata.index.name = "sample_id"
    v2_white_2014_metadata["ptb"] = 2
    for i in range(len(v2_white_2014_metadata.iloc[:,8])):
        if v2_white_2014_metadata.iloc[i,8] == "Term":
            v2_white_2014_metadata.iloc[i,12] = 0
        else:
            v2_white_2014_metadata.iloc[i,12] = 1

    v2_white_2014_metadata = v2_white_2014_metadata.sort_index()

    # apply log
    v2_white_2014 = np.log10(tmp.fillna(0))
    v2_white_2014 = v2_white_2014.replace(-np.inf, np.nan)

    # map the comp id
    v2_white_2014.columns = v2_white_2014.columns.to_series().map(v2_white_2014_cid)

    # save
    v2_white_2014.index.name = "sample_id"
    v2_white_2014 = v2_white_2014.loc[v2_white_2014_metadata["SAMPLE ID"].sort_values().index]
    v2_white_2014 = v2_white_2014.sort_index()

    v2_white_2014.columns = [str(c) for c in v2_white_2014.columns]

    return v2_white_2014, v2_white_2014_metadata