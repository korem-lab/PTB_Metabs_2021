# import bunch of fe functions and packages here
import math
import time
import shap
import numpy as np
import pandas as pd
from pandas import Series
from os.path import join, split, abspath
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA, KernelPCA
from pandas import Series, DataFrame, concat, to_pickle
from skbio.diversity.alpha import chao1, shannon, simpson
from scipy.stats import spearmanr, kruskal, pearsonr, mode
from sklearn.feature_selection import mutual_info_classif
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier

class FS_Method:
    KW = 'KruskalWallis'
    SP = 'Spearman'

class Alpha_Diversity:
    CHAO1 = 'chao1'
    SHANNON = 'shannon'
    SIMPSON = 'simpson'
    ALL = [CHAO1, SHANNON, SIMPSON]

class Transformations:
    NA = 'nothing'
    LAA = 'laa'

class PCA_Type:
    LINEAR = 'linear'
    POLYNOMIAL_KERNEL = 'polynomial'
    ALL = [LINEAR, POLYNOMIAL_KERNEL]

class Imp_Method:
    NA = 'nothing'
    ZERO = 'zero'
    MIN = 'min'
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'most_frequent'
    ALL = [ZERO, MIN, MEAN, MEDIAN, MODE]

class Std_Method:
    MEAN = 'mean'
    ROBUST_MEDIAN = 'robust_median'
    All = [MEAN, ROBUST_MEDIAN]

# Perform imputation for metabolomics data, can be used for other datasets upon modification.
def perform_imputate(x_train, x_test, method):

    if method == Imp_Method.NA:
        pass
    elif method == Imp_Method.ZERO:
        x_train = x_train.fillna(0)
        x_test = x_test.fillna(0)
    elif method ==Imp_Method.MIN:
        x_train = x_train.fillna(x_train.min())
        x_test = x_test.fillna(x_train.min())
    elif method ==Imp_Method.MEAN:
        x_train = x_train.fillna(x_train.mean())
        x_test = x_test.fillna(x_train.mean())
    elif method ==Imp_Method.MEDIAN:
        x_train = x_train.fillna(x_train.median())
        x_test = x_test.fillna(x_train.median())
    elif method ==Imp_Method.MODE:
        x_train = x_train.fillna(x_train.mode())
        x_test = x_test.fillna(x_train.mode())
    else:
        raise NotImplementedError("the imputation method passed is not implemented")

    if method != Imp_Method.NA:
        x_train = x_train.dropna(axis='columns')
        x_test = x_test[x_train.columns]

    return x_train, x_test

# Perform standardization, can be used for other datasets upon modification.
def perform_std(x_train, x_test, std_method, robust_quantile=0.05):#, robust, robust_quantile=0.05):

    x_old_train = x_train.copy() # save for x_test

    if std_method == Std_Method.MEAN:
        x_train = x_train.apply(lambda x: (x-np.nanmean(x.values))/np.nanstd(x.values), axis=0)
        x_test = x_test.apply(lambda y: (y-np.nanmean(x_old_train[y.name].values))/np.nanstd(x_old_train[y.name].values), axis=0)

    elif std_method == Std_Method.ROBUST_MEDIAN:
        #pds std is default to skip nan
        x_train = (x_train-x_train.median()) / (x_train[(x_train >= x_train.quantile(robust_quantile)) &\
                  (x_train <= x_train.quantile(1 - robust_quantile))]).std()
        x_test = (x_test-x_old_train.median()) / (x_test[(x_test >= x_old_train.quantile(robust_quantile)) &\
                  (x_test <= x_old_train.quantile(1 - robust_quantile))]).std()
    else:
        raise NotImplementedError("the standardization method passed is not implemented")

    return x_train, x_test

### PCA transformation
def get_pc_values(X_train, X_test, nPC, pca_method):
    '''
    Modified to try to add the PC values to the original feature set instead of
    replacing the original features
    '''

    if nPC > X_train.shape[0]:
        nPC = X_train.shape[0]
    if nPC > X_train.shape[1]:
        nPC = X_train.shape[1]

    nPC = int(nPC)
    if pca_method == PCA_Type.POLYNOMIAL_KERNEL:
        pca = KernelPCA(n_components=nPC, kernel="poly")
    else:
        pca = PCA(n_components=nPC)

    # fit only on the train
    train_index_pca = X_train.index
    test_index_pca = X_test.index

    try:
        pc_train = pd.DataFrame(pca.fit_transform(X_train), index=train_index_pca,
                               columns=(f'PC{n}' for n in range(1, nPC + 1)))
        pc_test = pd.DataFrame(pca.transform(X_test), index=test_index_pca, columns=(f'PC{n}' for n in range(1, nPC + 1)))
    except:
        print("PCA failed! Returning empty dataframes")
        return pd.DataFrame(), pd.DataFrame()

    return pc_train, pc_test

# perform absolute abundance transformations using total abundance from the metadata
def perform_abs_abundance(df, do_log, log_eps, metadata, train_index, test_index):

    load_mean = metadata.loc[metadata.index.isin(train_index)]["load"].mean()
    metadata["load"] = metadata["load"].fillna(load_mean)

    cols = df.columns
    rows = df.index
    df = df.values
    df = df / df.sum(axis=1, keepdims=True)

    df_new = pd.DataFrame(df, columns=cols, index=rows)
    df_new = df_new.mul(metadata["load"].loc[df_new.index], axis=0).fillna(0)

    if do_log:
        df_new = np.log10(df_new + log_eps)

    return df_new

# Add alpha diversity
def perform_diversity_adding(x_train, x_test, asv_counts, diversity_metric):
    if diversity_metric == Alpha_Diversity.SHANNON:
        func = shannon
    elif diversity_metric == Alpha_Diversity.SIMPSON:
        func = simpson
    else:
        func = chao1

    alpha = asv_counts.apply(func, axis=1)
    x_train = x_train.merge(alpha.rename('alpha'), how='left', left_index=True, right_index=True)
    x_test = x_test.merge(alpha.rename('alpha'), how='left', left_index=True, right_index=True)
    return x_train, x_test

# Spearman and KruskalWallis feature selection
def perform_feature_selection(X_train, X_test, fs_threshold, y_train, fs_method):

    if fs_method == FS_Method.KW:
        met = lambda x: kruskal(x, y_train.astype('category'))[1]
    else:
        met = lambda x: spearmanr(x, y_train)[1]

    selected_features = list(X_train.apply(met).sort_values(ascending=True)[:int(np.ceil(fs_threshold * len(X_train.columns)))].index)

    X_train_selected = X_train.loc[:,selected_features]
    X_test_selected = X_test.loc[:,selected_features]

    return X_train_selected, X_test_selected

# Use information gain to perform feature selection
def perform_info_gain(x_train, x_test, y_train, ig_threshold, random_state=42):

    feature_info = mutual_info_classif(x_train, y_train, random_state=0)
    ig = pd.DataFrame()
    ig['feature'] = x_train.columns
    ig['info'] = feature_info
    ig_sorted = ig.sort_values(by=['info'], ascending=False).loc[ig['info']!=0]

    if ig_sorted.empty:
        return x_train, x_test
    else:
        x_train = x_train.loc[:, ig_sorted['feature']]
        high_info_features = list(ig_sorted[:int(np.ceil(ig_threshold * len(x_train.columns)))]['feature'])

        x_train_selected = x_train.loc[:,high_info_features]
        x_test_selected = x_test.loc[:,high_info_features]

        return x_train_selected, x_test_selected

# Use shapley values to perform feature selection; use lightGBM to generate shapley values
def perform_shap_selection(x_train, x_test, y_train, shap_threshold, ml_method, classifier, params, fe_hyper_param_keys, seed):


    # Create lightGBM model
    xgb_param = {}
    xgb_param.update(seed)

    if classifier:
        xgb_model = LGBMClassifier(**xgb_param) # defining our classifer based on params
    else:
        xgb_model = LGBMRegressor(**xgb_param)

    # do some clean up to avoid model warnings.
    params = {key: params[key] for key in params if key not in fe_hyper_param_keys}

    xgb_model.set_params(**params)
    xgb_model.fit(x_train, y_train.values.ravel())

    # Get shapley values from the model
    shap_values_train = shap.TreeExplainer(xgb_model).shap_values(x_train)
    if classifier:
        shap_abs_sum_sorted = pd.DataFrame(data=[np.abs(i).sum(axis=0)\
        for i in shap_values_train], columns=x_train.columns).sum().sort_values(ascending=False)
    else:
        shap_abs_sum_sorted = pd.DataFrame(data=np.abs(shap_values_train),\
        columns=x_train.columns).sum().sort_values(ascending=False)

    shap_abs_sum_sorted = shap_abs_sum_sorted[shap_abs_sum_sorted != 0]

    if shap_abs_sum_sorted.empty:
        return x_train, x_test
    else:
        x_train = x_train.loc[:,shap_abs_sum_sorted.index]

        # Select the features with the most in the sum of the magnitude of shaps across samples
        high_shap_features = list(shap_abs_sum_sorted[:int(np.ceil(shap_threshold * len(x_train.columns)))].index)
        x_train_selected = x_train.loc[:,high_shap_features]
        x_test_selected = x_test.loc[:,high_shap_features]

        return x_train_selected, x_test_selected

# Filter out features that have more than the accepted amount of empty/zero values set by the threshold
def perform_feature_filtering(X_train, X_test, ff_threshold):
    # because of transformation, the value in dataset should not contain Nan, but if it is, filter will do nothing because comparison
    # of nan to float always return false.

    modes = list(X_train.mode().iloc[0,:])
    filtered_features = list((X_train != modes).sum().sort_values(ascending=False)[:int(np.ceil(ff_threshold * len(X_train.columns)))].index)
    X_train_filtered = X_train.loc[:, filtered_features]
    X_test_filtered = X_test.loc[:, filtered_features]

    return X_train_filtered, X_test_filtered

def fe_pipe(do_imputate, imputate_method, do_standardize, std_method, transformation, do_pca, pca_method, nPC, do_feature_selection,
            fs_threshold, fs_method, do_feature_filtering, ff_threshold, do_shap_selection, shap_threshold, do_info_gain, ig_threshold,
            do_diversity_adding, diversity_metric, do_total_reads_adding, do_ycap, do_remove_from_train, train_samples_to_remove,
            fe_hyper_param_keys, train_index, test_index, x, y, metadata, external_validation, ev_test, ev_y, shap_importance, combination, params,
            classifier, ml_method, seed, log_eps):

    # make sure there exists at least two unique values for each column
    x = x.loc[:, x.nunique() > 1].copy()
    df = x.copy()
    x_copy = x.copy()
    metadata = metadata.copy()

    if combination:
        df = df.loc[:, [col for col in df.columns if 'micro' in col]]

    if ev_test is not None:
        ev_df = ev_test.copy(deep=True)
    else:
        ev_df = ev_test

    if transformation == Transformations.LAA:
        df = perform_abs_abundance(df, True, log_eps, metadata, train_index, test_index)
        if ev_df is not None:
            ev_df = perform_abs_abundance(ev_df, True, log_eps, metadata, train_index, test_index)

    if combination:
        df_others = x.copy(deep=True)
        df_others = df_others.loc[:, [col for col in df_others.columns if 'micro' not in col]]
        df = pd.concat([df, df_others], axis=1)
        df.index.names = ['sample_id']

    # Select/split samples according to the cv definition defined in define_cv_list()
    x_train, x_test = df.loc[[t for t in train_index if t in df.index]], df.loc[[t for t in test_index if t in df.index]]
    y_train, y_test = y.loc[[t for t in train_index if t in df.index]], y.loc[[t for t in test_index if t in df.index]]

    # make sure there exists at least two unique values for each column
    x_train = x_train.loc[:, x_train.nunique() > 1]
    x_test = x_test[x_train.columns]

    if do_remove_from_train:
        # to_be_dropped = list(set(x_train.index) & set(train_samples_to_remove.index))
        to_be_dropped = list(set(x_train.index) & set(train_samples_to_remove))

        x_train = x_train.drop(to_be_dropped)
        y_train = y_train.drop(to_be_dropped)

    # This is to assymetrically match ev columns with df columns, also assign test data
    if external_validation:
        x_test = ev_df
        y_test = ev_y

        relevant_columns = []
        for i in list(x_test.columns):
            if i in list(x_train.columns):
                relevant_columns.append(i)
        x_test = x_test.loc[:,relevant_columns]

        empty_columns = []
        for i in x_train:
            if i not in x_test:
                empty_columns.append(i)
        x_test = pd.concat([x_test, pd.DataFrame(columns=empty_columns)])
        x_test = x_test.replace({0:np.nan})
        x_test.index.name = 'sample_id'

    if combination:
        x_train_copy, x_test_copy = x_train.copy(deep=True), x_test.copy(deep=True)
        x_train = x_train.loc[:, [col for col in x_train.columns if 'metab' in col]]
        x_test = x_test.loc[:, [col for col in x_test.columns if 'metab' in col]]

    # standardize or normalize, if both are set to true, do standardization.
    if do_standardize:
        x_train, x_test = perform_std(x_train, x_test, std_method)
        x_train = x_train.loc[:, x_train.nunique() > 1]
        x_test = x_test[x_train.columns]

    if combination:
        x_train_others = x_train_copy.loc[:, [col for col in x_train_copy.columns if 'metab' not in col]]
        x_train = pd.concat([x_train, x_train_others], axis=1)
        x_train.index.names = ['sample_id']

        x_test_others = x_test_copy.loc[:, [col for col in x_test_copy.columns if 'metab' not in col]]
        x_test = pd.concat([x_test, x_test_others], axis=1)
        x_test.index.names = ['sample_id']

    # imputate data
    if do_imputate:
        x_train, x_test = perform_imputate(x_train, x_test, imputate_method)

    if combination:
        x_train_copy, x_test_copy = x_train.copy(deep=True), x_test.copy(deep=True)
        x_train = x_train.loc[:, [col for col in x_train.columns if 'micro' in col]]
        x_test = x_test.loc[:, [col for col in x_test.columns if 'micro' in col]]

    old_xt, old_xts = x_train.copy(), x_test.copy()
    if do_pca: # Try adding pc's of the original features to the clustered features
        x_train, x_test = get_pc_values(old_xt, old_xts, nPC, pca_method)

    if combination:
        x_train_others = x_train_copy.loc[:, [col for col in x_train_copy.columns if 'micro' not in col]]
        x_train = pd.concat([x_train, x_train_others], axis=1)
        x_train.index.names = ['sample_id']

        x_test_others = x_test_copy.loc[:, [col for col in x_test_copy.columns if 'micro' not in col]]
        x_test = pd.concat([x_test, x_test_others], axis=1)
        x_test.index.names = ['sample_id']


    if do_feature_filtering:
        x_train, x_test = perform_feature_filtering(x_train, x_test, ff_threshold)

    if do_shap_selection:
        x_train, x_test = perform_shap_selection(x_train, x_test, y_train, shap_threshold, ml_method, classifier, params, fe_hyper_param_keys, seed)

    if do_info_gain:
        x_train, x_test = perform_info_gain(x_train, x_test, y_train, ig_threshold)

    if do_feature_selection:
        x_train, x_test = perform_feature_selection(x_train, x_test, fs_threshold, y_train, fs_method)

    if do_total_reads_adding:
        for x in (x_train, x_test):
            x['total_reads'] = x.sum(axis=1)

    if do_diversity_adding:
        x_train, x_test = perform_diversity_adding(x_train, x_test, x_copy, diversity_metric)

    if do_ycap is not None:
        y_train = y_train.clip(upper = do_ycap)

    x_train = x_train.dropna(axis='columns')
    x_test = x_test[x_train.columns]

    return x_train, y_train, x_test, y_test
