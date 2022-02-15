import pyreadr
import numpy as np
import pandas as pd
from pandas import Series
import sys
from os.path import join, exists
from pandas import read_pickle, read_excel

from definitions import elovitz_ct_path, md_path, metabs_path, NMPCs_path


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

    return y


def get_metabs(rzscore=True, min_pres=5, min_impute=True):

    def _fmin(d):
        return d.fillna(d.min())

    def _rzsc(d, q = 0.05):
        return (d-d.median()) / (d[(d >= d.quantile(q)) & (d <= d.quantile(1-q))]).std()

    ret = pd.read_csv(metabs_path, index_col=0)
    ret = ret.drop('Volume extracted (ul)', axis=1)

    if rzscore:
        ret = _rzsc(ret.apply(np.log10))

    if min_impute:
        ret = _fmin(ret)

    if min_pres > 0:
        ret = ret.loc[:, (ret > ret.min()).sum() > min_pres]

    return ret


def get_md(metabs=True):

    md = pd.read_pickle(md_path)
    md.index = md.index.to_series().apply(lambda v: f'{int(v):04}.V2')

    ## remove iPTB/mPTB samples
    md = md[md.PTB.isin({0,1})]

    if metabs:
        m = get_metabs()
        md = md.loc[m.index]

    md['PTB'] = md.PTB.replace(0, 'TB').replace(1, 'sPTB')
    md['PTB'] = md['PTB'].astype(str)

    md = md.drop(['cst at v2'], axis=1)
    md.rename(columns={'cst':'OrigCST'}, inplace=True)

    return md


def get_tym_nmpcs(metabs=True):

    NMPCs = pd.read_csv(NMPCs_path, index_col=0)

    eps = NMPCs[NMPCs > 0].min(axis=1).sort_values().iloc[0] / 10
    ## use log(x+1) transform to reduce skew while preserving rank
    NMPCs= np.log10(NMPCs + eps)

    return NMPCs


if __name__ == '__main__':
    md = get_md()
    mt = get_metabs()
    ct = get_elovitz_cts_1k()
    assert (md.index == mt.index).all() and (ct.index == mt.index).all()
