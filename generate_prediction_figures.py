import glob
import os
import pickle
import statistics
import numpy as np
import pandas as pd
from os.path import join, exists
from pandas import Series, DataFrame, concat, to_pickle
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.training_pipe import pipe, Study_Design
from utils.machine_learning import Machine_Learning_Method
from utils.accuracy_evaluation import eval_nest, eval_nest_rs, eval_external, eval_benchmark, eval_benchmark_cat, eval_race_val, eval_ns_to_s
from utils.prediction_figures import main_auroc, main_auprc, gw_thresh, external_auroc, external_auprc, micro_shap, metab_shap, combo_shap,\
SV_LR_comparison, race_strat_compare_to_non_strat, model_performance_by_race, batch_auroc

from utils.get_dfs import get_md, get_sample_batches, get_metabs, get_cid_di, get_usearch_otus, get_clin_df_for_pred, get_ghartey_2014, get_ghartey_2016

import warnings
warnings.filterwarnings('ignore')


def main():
    '''
    Pre-process, save, and load data.
    '''
    print ("Pre-processing data...")
    metadata = get_md()
    metadata = metadata.rename(columns={'delGA':'gw', 'bLoad':'load'})
    batches = get_sample_batches()
    metadata = metadata.join(batches)
    metadata_AA = metadata.loc[metadata['race']==1]
    metadata_non_AA = metadata.loc[metadata['race'].isin([0,2])]
    metadata_batch_1 = metadata.loc[metadata['LC MS/MS Pos Early, Pos Late, Polar Batch'] == 'batch_1']
    metadata_batch_2 = metadata.loc[metadata['LC MS/MS Pos Early, Pos Late, Polar Batch'] == 'batch_2']

    clinical = get_clin_df_for_pred()

    microbiome = get_usearch_otus() ## rename this function and move to utils.get_dfs
    microbiome_AA = microbiome.loc[metadata_AA.index]
    microbiome_non_AA = microbiome.loc[metadata_non_AA.index]

    metabolomics = np.log10(get_metabs(min_impute=False, rzscore=False, min_pres=0, compids=True))
    metabolomics_AA = metabolomics.loc[metadata_AA.index]
    metabolomics_non_AA = metabolomics.loc[metadata_non_AA.index]
    metabolomics_batch_1 = metabolomics.loc[metadata_batch_1.index]
    metabolomics_batch_2 = metabolomics.loc[metadata_batch_2.index]

    combination = pd.concat([clinical.add_suffix("_clin"), microbiome.add_suffix('_micro'), metabolomics.add_suffix('_metab')], axis=1)
    combination.index.name = 'sample_id'
    combination_AA = combination.loc[metadata_AA.index]
    combination_non_AA = combination.loc[metadata_non_AA.index]

    train_samples_to_remove = list(metadata[metadata.race == 2].index)

    if not os.path.exists("data/v2_2016.csv"):
        v2_2016, v2_2016_md = get_ghartey_2016()
        v2_2016.to_csv("data/v2_2016.csv")
        v2_2016_md.to_csv("data/v2_2016_md.csv")
    v2_2016 = pd.read_csv("data/v2_2016.csv", index_col=0)
    v2_2016_md = pd.read_csv("data/v2_2016_md.csv", index_col=0)

    if not os.path.exists("data/v2_2014.csv"):
        v2_2014, v2_2014_md = get_ghartey_2014()
        v2_2014.to_csv("data/v2_2014.csv")
        v2_2014_md.to_csv("data/v2_2014_md.csv")
    v2_2014 = pd.read_csv("data/v2_2014.csv", index_col=0)
    v2_2014_md = pd.read_csv("data/v2_2014_md.csv", index_col=0)

    data_list = [clinical, microbiome_AA, microbiome_non_AA, metabolomics_AA, metabolomics_non_AA, combination_AA, combination_non_AA]
    md_list = [metadata.copy(), metadata_AA.copy(), metadata_non_AA.copy(), metadata_AA.copy(),
               metadata_non_AA.copy(), metadata_AA.copy(), metadata_non_AA.copy()]

    data_benchmark_list = [metabolomics_AA, metabolomics_non_AA, metabolomics_AA, metabolomics_non_AA, metabolomics_AA, metabolomics_non_AA,
                           metabolomics]
    md_benchmark_list = [metadata_AA.copy(), metadata_non_AA.copy(), metadata_AA.copy(),
                         metadata_non_AA.copy(), metadata_AA.copy(), metadata_non_AA.copy(), metadata.copy()]

    data_race_val_list = [metabolomics_non_AA, metabolomics_AA]
    md_race_val_list = [metadata_non_AA.copy(), metadata_AA.copy()]

    data_batch_list = [metabolomics_batch_1, metabolomics_batch_2]
    md_batch_list = [metadata_batch_1.copy(), metadata_batch_2.copy()]

    ev_data_list = [v2_2016, v2_2014]
    ev_md_list = [v2_2016_md, v2_2014_md]

    # otus_sintax = pd.read_csv('../data/other/otus.sintax', sep = '\t', header = None).set_index(0)[3]\
    otus_sintax = pd.read_csv('data/otus.sintax', sep = '\t', header = None).set_index(0)[3]\
    .apply(lambda x: [t.split(':')[1] for t in x.split(',')] if not pd.isnull(x) else np.nan).apply(Series)\
    .rename(columns = {0:'Kingdom', 1:'Phylum', 2:'Class', 3:'Order', 4:'Family', 5:'Genus', 6:'Species'})
    
    metab_ID = get_cid_di()


    '''
    Load models.
    '''
    print ("Loading models...")
    clinical_ehps_list = [pd.read_pickle('ml_models/clinical_nested/best_models/'+ str(j+1)) for j in range(50)]
    clinical_cv = [pd.read_pickle(i) for i in sorted(glob.glob(join('ml_models/clinical_nested', 'nested_structure/*')))]

    microbiome_AA_ehps_list = [pd.read_pickle('ml_models/microbiome_nested/AA/best_models/'+ str(j+1)) for j in range(50)]
    microbiome_AA_cv = [pd.read_pickle(i) for i in sorted(glob.glob(join('ml_models/microbiome_nested/AA', 'nested_structure/*')))]
    microbiome_AA_shap_model = pd.read_pickle('ml_models/shap_models/micro_AA')
    microbiome_non_AA_ehps_list = [pd.read_pickle('ml_models/microbiome_nested/non_AA/best_models/'+ str(j+1)) for j in range(50)]
    microbiome_non_AA_cv = [pd.read_pickle(i) for i in sorted(glob.glob(join('ml_models/microbiome_nested/non_AA', 'nested_structure/*')))]
    microbiome_non_AA_shap_model = pd.read_pickle('ml_models/shap_models/micro_non_AA')

    metabolomics_AA_ehps_list = [pd.read_pickle('ml_models/metabolomics_nested/AA/best_models/'+ str(j+1)) for j in range(50)]
    metabolomics_AA_cv = [pd.read_pickle(i) for i in sorted(glob.glob(join('ml_models/metabolomics_nested/AA', 'nested_structure/*')))]
    metabolomics_AA_shap_model = pd.read_pickle('ml_models/shap_models/metab_AA')
    metabolomics_non_AA_ehps_list = [pd.read_pickle('ml_models/metabolomics_nested/non_AA/best_models/'+ str(j+1)) for j in range(50)]
    metabolomics_non_AA_cv = [pd.read_pickle(i) for i in sorted(glob.glob(join('ml_models/metabolomics_nested/non_AA', 'nested_structure/*')))]
    metabolomics_non_AA_shap_model = pd.read_pickle('ml_models/shap_models/metab_non_AA')

    combination_AA_ehps_list = [pd.read_pickle('ml_models/combination_nested/AA/best_models/'+ str(j+1)) for j in range(50)]
    combination_AA_cv = [pd.read_pickle(i) for i in sorted(glob.glob(join('ml_models/combination_nested/AA', 'nested_structure/*')))]
    combination_AA_shap_model = pd.read_pickle('ml_models/shap_models/combo_AA')
    combination_non_AA_ehps_list = [pd.read_pickle('ml_models/combination_nested/non_AA/best_models/'+ str(j+1)) for j in range(50)]
    combination_non_AA_cv = [pd.read_pickle(i) for i in sorted(glob.glob(join('ml_models/combination_nested/non_AA', 'nested_structure/*')))]
    combination_non_AA_shap_model = pd.read_pickle('ml_models/shap_models/combo_AA')

    ehps_list = [clinical_ehps_list, microbiome_AA_ehps_list, microbiome_non_AA_ehps_list, metabolomics_AA_ehps_list, metabolomics_non_AA_ehps_list,
                 combination_AA_ehps_list, combination_non_AA_ehps_list]
    shap_ehps_list = [microbiome_AA_shap_model, microbiome_non_AA_shap_model, metabolomics_AA_shap_model, metabolomics_non_AA_shap_model,
                      combination_AA_shap_model, combination_non_AA_shap_model]
    cv_list = [clinical_cv, microbiome_AA_cv, microbiome_non_AA_cv, metabolomics_AA_cv, metabolomics_non_AA_cv,
               combination_AA_cv, combination_non_AA_cv]

    lightGBM_AA_hp = pd.read_pickle('ml_models/benchmark_models/benchmark_lightGBM_AA')
    lightGBM_AA_cv = pd.read_pickle('ml_models/benchmark_models/benchmark_lightGBM_AA_cv')

    lightGBM_non_AA_hp = pd.read_pickle('ml_models/benchmark_models/benchmark_lightGBM_non_AA')
    lightGBM_non_AA_cv = pd.read_pickle('ml_models/benchmark_models/benchmark_lightGBM_non_AA_cv')

    SVR_AA_hp = pd.read_pickle('ml_models/benchmark_models/benchmark_SVR_AA')
    SVR_AA_cv = pd.read_pickle('ml_models/benchmark_models/benchmark_SVR_AA_cv')

    SVR_non_AA_hp = pd.read_pickle('ml_models/benchmark_models/benchmark_SVR_non_AA')
    SVR_non_AA_cv = pd.read_pickle('ml_models/benchmark_models/benchmark_SVR_non_AA_cv')

    LR_AA_hp = pd.read_pickle('ml_models/benchmark_models/benchmark_LR_AA')
    LR_AA_cv = pd.read_pickle('ml_models/benchmark_models/benchmark_LR_AA_cv')

    LR_non_AA_hp = pd.read_pickle('ml_models/benchmark_models/benchmark_LR_non_AA')
    LR_non_AA_cv = pd.read_pickle('ml_models/benchmark_models/benchmark_LR_non_AA_cv')

    metab_ns_hp = pd.read_pickle('ml_models/benchmark_models/benchmark_metab_ns')
    metab_ns_cv = pd.read_pickle('ml_models/benchmark_models/benchmark_metab_ns_cv')

    batch_1_hp = pd.read_pickle('ml_models/batch_models/batch_1')
    batch_2_hp = pd.read_pickle('ml_models/batch_models/batch_2')
    batch_1_cv = pd.read_pickle('ml_models/batch_models/batch_1_cv')
    batch_2_cv = pd.read_pickle('ml_models/batch_models/batch_2_cv')

    benchmark_ehps_list = [lightGBM_AA_hp, lightGBM_non_AA_hp, SVR_AA_hp, SVR_non_AA_hp, LR_AA_hp, LR_non_AA_hp, metab_ns_hp]
    benchmark_cv_list = [lightGBM_AA_cv, lightGBM_non_AA_cv, SVR_AA_cv, SVR_non_AA_cv, LR_AA_cv, LR_non_AA_cv, metab_ns_cv]

    batch_ehps_list = [batch_1_hp, batch_2_hp]
    batch_cv_list = [batch_1_cv, batch_2_cv]

    random_sampling_acc = pd.read_pickle('ml_models/random_sampling_acc')

    '''
    Load study designs.
    '''
    print ("Loading study designs...")
    study_design_list = [Study_Design.NORMAL_OPTIMIZATION, Study_Design.NESTED_OPTIMIZATION, Study_Design.RANDOM_SAMPLING_OPTIMIZATION,
                         Study_Design.CHECK_SHAP_IMPORTANCE, Study_Design.NESTED_VALIDATION, Study_Design.EXTERNAL_VALIDATION,
                         Study_Design.NESTED_P_VALUE]
    ml_method_list = [Machine_Learning_Method.LIGHTGBM, Machine_Learning_Method.SVR, Machine_Learning_Method.LR]
    ml_method_benchmark_list = [Machine_Learning_Method.LIGHTGBM, Machine_Learning_Method.LIGHTGBM,
                                Machine_Learning_Method.SVR, Machine_Learning_Method.SVR,
                                Machine_Learning_Method.LR, Machine_Learning_Method.LR,
                                Machine_Learning_Method.LIGHTGBM]



    '''
    Train the models
    '''
    if exists('ml_models/res_nested'):
        print ("Saved main nested models detected, unpacking...")
        result_robust_list = pd.read_pickle('ml_models/res_nested')
    else:
        result_robust_list = []
        print ("Training main nested models, " + str(len(ehps_list)) + " models in total...")
        for i in range(len(ehps_list)):
            print ("Training main nested models: " + str(i+1) + "...")
            result = []
            result_list = []
            stop_count = 0
            for j in range(len(ehps_list[i])):
                stop_count += 1
                result.append(pipe(X = data_list[i].copy(),
                                   md = md_list[i].copy(),
                                   params_list = ehps_list[i][j].copy(),
                                   cv = cv_list[i][j].copy(),
                                   classifier = False,
                                   ml_method = ml_method_list[0],
                                   study_design = study_design_list[4],
                                   train_samples_to_remove = train_samples_to_remove))
                if stop_count == 10:
                    print (str(len(ehps_list[i])-(j+1)) + " more models remaining...")
                    result_list.append(result)
                    result = []
                    stop_count = 0
            result_robust_list.append(result_list)
        # Attain results
        with open('ml_models/res_nested', 'wb') as f:
            pickle.dump(result_robust_list, f)



    '''
    Train the benchmark models
    '''
    if exists('ml_models/res_benchmark'):
        print ("Saved benchmark models detected, unpacking...")
        result_benchmark_list = pd.read_pickle('ml_models/res_benchmark')
    else:
        result_benchmark_list = []
        print ("Training benchmark models, " + str(len(benchmark_ehps_list)) + " models in total...")
        for i in range(len(benchmark_ehps_list)):
            print ("Training benchmark models: " + str(i+1) + "...")
            result = []
            result_list = []
            stop_count = 0
            for j in range(len(benchmark_cv_list[i])):
                stop_count += 1
                result.append(pipe(X = data_benchmark_list[i].copy(),
                                   md = md_benchmark_list[i].copy(),
                                   params_list = benchmark_ehps_list[i].copy(),
                                   cv = benchmark_cv_list[i][j].copy(),
                                   classifier = False,
                                   ml_method = ml_method_benchmark_list[i],
                                   study_design = study_design_list[0],
                                   train_samples_to_remove = train_samples_to_remove))
                if stop_count == 10:
                    print (str(len(benchmark_cv_list[i])-(j+1)) + " more models remaining...")
                    result_list.append(result)
                    result = []
                    stop_count = 0
            result_benchmark_list.append(result_list)
        # Attain results
        with open('ml_models/res_benchmark', 'wb') as f:
            pickle.dump(result_benchmark_list, f)
    result_benchmark_list_cat = [[result_benchmark_list[0], result_benchmark_list[1]],
                                 [result_benchmark_list[2], result_benchmark_list[3]],
                                 [result_benchmark_list[4], result_benchmark_list[5]]]



    '''
    Validate benchmark models across race
    '''
    if exists('ml_models/res_race_val'):
        print ("Saved race validation models detected, unpacking...")
        result_race_val_list = pd.read_pickle('ml_models/res_race_val')
    else:
        result_race_val_list = []
        print ("Validating benchmark models across race...")
        for i in range(2):
            result_race_val_list.append(pipe(X = data_benchmark_list[i].copy(),
                                             md = md_benchmark_list[i].copy(),
                                             params_list = benchmark_ehps_list[i].copy(),
                                             cv = {"1_KFold_0":[list(data_benchmark_list[i].index), list(data_race_val_list[i].index)]},
                                             classifier = False,
                                             ml_method = ml_method_list[0],
                                             study_design = study_design_list[5],
                                             train_samples_to_remove = train_samples_to_remove,
                                             ev_data=data_race_val_list[i],
                                             ev_md=md_race_val_list[i]))
        # Attain results
        with open('ml_models/res_race_val', 'wb') as f:
            pickle.dump(result_race_val_list, f)



    '''
    Validate the metabolomics models
    '''
    if exists('ml_models/res_validation'):
        print ("Saved metabolomics validation models detected, unpacking...")
        result_validation_list = pd.read_pickle('ml_models/res_validation')
    else:
        print ("Validating the metabolomics models...")
        result_validation_list = []
        for i in range(2):
            result_validation_list.append(pipe(X = data_list[i+3].copy(),
                                               md = md_list[i+3].copy(),
                                               params_list = shap_ehps_list[i+2].copy(),
                                               cv = {"1_KFold_0":[list(data_list[i+3].index), list(data_list[i+3].index)]},
                                               classifier = False,
                                               ml_method = ml_method_list[0],
                                               study_design = study_design_list[5],
                                               train_samples_to_remove = train_samples_to_remove,
                                               ev_data=ev_data_list[i],
                                               ev_md=ev_md_list[i]))
        # Attain results
        with open('ml_models/res_validation', 'wb') as f:
            pickle.dump(result_validation_list, f)



    '''
    Investigate batch effect
    '''
    if exists('ml_models/res_batch'):
        print ("Saved batch effect models detected, unpacking...")
        result_batch_list = pd.read_pickle('ml_models/res_batch')
    else:
        result_batch_list = []
        print ("Training batch effect models, 2 models in total...")
        for i in range(2):
            print ("Training batch effects models: " + str(i+1) + "...")
            result = []
            result_list = []
            stop_count = 0
            for j in range(50):
                stop_count += 1
                result.append(pipe(X = data_batch_list[i].copy(),
                                   md = md_batch_list[i].copy(),
                                   params_list = batch_ehps_list[i].copy(),
                                   cv = batch_cv_list[i][j],
                                   classifier = False,
                                   ml_method = ml_method_list[0],
                                   study_design = study_design_list[0],
                                   train_samples_to_remove = train_samples_to_remove,
                                   ev_data= None,
                                   ev_md = None))
                if stop_count == 10:
                    print (str(50-(j+1)) + " more models remaining...")
                    result_list.append(result)
                    result = []
                    stop_count = 0
            result_batch_list.append(result_list)

        for i in range(2):
            result_batch_list.append(pipe(X = data_batch_list[i].copy(),
                                          md = md_batch_list[i].copy(),
                                          params_list = batch_ehps_list[i].copy(),
                                          cv = {"1_KFold_0":[list(data_batch_list[i].index), list(data_batch_list[-i].index)]},
                                          classifier = False,
                                          ml_method = ml_method_list[0],
                                          study_design = study_design_list[5],
                                          train_samples_to_remove = train_samples_to_remove,
                                          ev_data=data_batch_list[1-i],
                                          ev_md=md_batch_list[1-i]))
        # Attain results
        with open('ml_models/res_batch', 'wb') as f:
            pickle.dump(result_batch_list, f)



    '''
    Generate the shapley values
    '''
    if not os.path.exists("shap_tables"):
        os.makedirs("shap_tables")

    if exists('ml_models/res_shap'):
        print ("Saved main shap models detected, unpacking...")
        result_shap_list = pd.read_pickle('ml_models/res_shap')
    else:
        print ("Generating the shapley values...")
        result_shap_list = []
        for i in range(6):
            result_shap_list.append(pipe(X = data_list[i+1].copy(),
                                         md = md_list[i+1].copy(),
                                         params_list = shap_ehps_list[i].copy(),
                                         cv = {"1_KFold_0":[list(data_list[i+1].index), list(data_list[i+1].index)]},
                                         classifier = False,
                                         ml_method = ml_method_list[0],
                                         study_design = study_design_list[3],
                                         train_samples_to_remove = train_samples_to_remove))
        # Attain results
        with open('ml_models/res_shap', 'wb') as f:
            pickle.dump(result_shap_list, f)



    '''
    Evaluate the models
    '''
    print ("Evaluating models...")
    clin_28, clin_32, clin_37, clin_pred = eval_nest(result_robust_list[0])

    micro_28, micro_32, micro_37, micro_pred = eval_nest_rs([result_robust_list[1],result_robust_list[2]])
    micro_28_AA, micro_32_AA, micro_37_AA, micro_AA_pred = eval_nest(result_robust_list[1])
    micro_28_non_AA, micro_32_non_AA, micro_37_non_AA, micro_non_AA_pred = eval_nest(result_robust_list[2])

    metab_28, metab_32, metab_37, metab_pred = eval_nest_rs([result_robust_list[3],result_robust_list[4]])
    metab_28_AA, metab_32_AA, metab_37_AA, metab_AA_pred = eval_nest(result_robust_list[3])
    metab_28_non_AA, metab_32_non_AA, metab_37_non_AA, metab_non_AA_pred = eval_nest(result_robust_list[4])

    combo_28, combo_32, combo_37, combo_pred = eval_nest_rs([result_robust_list[5],result_robust_list[6]])
    combo_28_AA, combo_32_AA, combo_37_AA, combo_AA_pred = eval_nest(result_robust_list[5])
    combo_28_non_AA, combo_32_non_AA, combo_37_non_AA, combo_non_AA_pred = eval_nest(result_robust_list[6])

    y_test_all, y_pred_all, auROC_all = eval_benchmark(result_benchmark_list)
    y_test_all_cat, y_pred_all_cat, auROC_all_cat = eval_benchmark_cat(result_benchmark_list_cat)

    y_test_AA_val_non_AA, y_pred_AA_val_non_AA, AA_val_non_AA_auROC = eval_race_val(result_race_val_list[0][0][0])
    y_test_non_AA_val_AA, y_pred_non_AA_val_AA, non_AA_val_AA_auROC = eval_race_val(result_race_val_list[1][0][0])

    y_test_met6_AA, y_pred_met6_AA, y_test_met6_non_AA, y_pred_met6_non_AA, \
    auROC_met6_AA, auROC_met6_non_AA = eval_ns_to_s(result_benchmark_list[6], list(metadata_AA.index), list(metadata_non_AA.index))

    v2_2016_test, v2_2016_pred = eval_external(result_validation_list[0])
    v2_2014_test, v2_2014_pred = eval_external(result_validation_list[1])

    batch_y_test_all, batch_y_pred_all, batch_auROC_all = eval_benchmark([result_batch_list[0], result_batch_list[1]])
    batch_1_val_test, batch_1_val_pred = eval_external(result_batch_list[2], False)
    batch_2_val_test, batch_2_val_pred = eval_external(result_batch_list[3], False)
    batch_1_val = roc_auc_score(batch_1_val_test, batch_1_val_pred)
    batch_2_val = roc_auc_score(batch_2_val_test, batch_2_val_pred)

    test_list_all_28 = [clin_28, micro_28, metab_28, combo_28]
    test_list_all_32 = [clin_32, micro_32, metab_32, combo_32]
    test_list_all_37 = [clin_37, micro_37, metab_37, combo_37]
    test_list_all_28_AA = [micro_28_AA, metab_28_AA]
    test_list_all_32_AA = [micro_32_AA, metab_32_AA]
    test_list_all_37_AA = [micro_37_AA, metab_37_AA]

    test_list_all = [test_list_all_28, test_list_all_32, test_list_all_37]
    test_list_all_AA = [test_list_all_28_AA, test_list_all_32_AA, test_list_all_37_AA]
    pred_list_all = [clin_pred, micro_pred, metab_pred, combo_pred]
    pred_list_all_AA = [micro_AA_pred, metab_AA_pred]

    auroc_list_all_37 = [[roc_auc_score(test_list_all_37[j][i], pred_list_all[j][i]) for i in range(5)] for j in range(4)]
    auroc_list_all_37_AA = [[roc_auc_score(test_list_all_37_AA[j][i], pred_list_all_AA[j][i]) for i in range(5)] for j in range(2)]
    auroc_list_all_32 = [[roc_auc_score(test_list_all_32[j][i], pred_list_all[j][i]) for i in range(5)] for j in range(4)]
    auroc_list_all_32_AA = [[roc_auc_score(test_list_all_32_AA[j][i], pred_list_all_AA[j][i]) for i in range(5)] for j in range(2)]
    auroc_list_all_28 = [[roc_auc_score(test_list_all_28[j][i], pred_list_all[j][i]) for i in range(5)] for j in range(4)]
    auroc_list_all_28_AA = [[roc_auc_score(test_list_all_28_AA[j][i], pred_list_all_AA[j][i]) for i in range(5)] for j in range(2)]
    auroc_list_all = [auroc_list_all_28, auroc_list_all_32, auroc_list_all_37]
    auroc_list_all_AA = [auroc_list_all_28_AA, auroc_list_all_32_AA, auroc_list_all_37_AA]
    auprc_list_all = [[average_precision_score(test_list_all_37[j][i], pred_list_all[j][i]) for i in range(5)] for j in range(4)]

    y_test_r, y_pred_r = [y_test_all[0], y_test_all[1], y_test_met6_AA, y_test_met6_non_AA],\
                           [y_pred_all[0], y_pred_all[1], y_pred_met6_AA, y_pred_met6_non_AA]
    auROC_r = [auROC_all[0], auROC_all[1], auROC_met6_AA, auROC_met6_non_AA]

    y_test_nr, y_pred_nr = [y_test_AA_val_non_AA, y_test_non_AA_val_AA], [y_pred_AA_val_non_AA, y_pred_non_AA_val_AA]
    auROC_nr = [AA_val_non_AA_auROC, non_AA_val_AA_auROC]

    '''
    Generate the figures
    '''
    print ("Generating the figures...")
    main_auroc(test_list_all_37, pred_list_all, auroc_list_all_37, False)
    main_auroc([test_list_all_37[2],test_list_all_37[3]], [pred_list_all[2], pred_list_all[3]], [auroc_list_all_37[2],auroc_list_all_37[3]], True)
    main_auprc(test_list_all_37, pred_list_all, auprc_list_all, False)
    main_auprc([test_list_all_37[2],test_list_all_37[3]], [pred_list_all[2], pred_list_all[3]], [auprc_list_all[2],auprc_list_all[3]], True)
    gw_thresh(test_list_all_AA, pred_list_all_AA, auroc_list_all_AA, True)
    gw_thresh(test_list_all_AA, pred_list_all_AA, auroc_list_all_AA, False)
    external_auroc(v2_2016_test, v2_2016_pred, v2_2014_test, v2_2014_pred)
    external_auprc(v2_2016_test, v2_2016_pred, v2_2014_test, v2_2014_pred)
    micro_shap(result_shap_list[0], result_shap_list[1], otus_sintax)
    metab_shap(result_shap_list[2], result_shap_list[3], metab_ID)
    combo_shap(result_shap_list[4], result_shap_list[5], metab_ID)
    SV_LR_comparison(y_test_all_cat, y_pred_all_cat, auROC_all_cat)
    race_strat_compare_to_non_strat(y_test_all_cat, y_pred_all_cat, auROC_all_cat, y_test_all, y_pred_all, auROC_all)
    model_performance_by_race(y_test_r, y_pred_r, auROC_r, y_test_nr, y_pred_nr, auROC_nr)
    batch_auroc(random_sampling_acc, statistics.mean(batch_auROC_all[0]), statistics.mean(batch_auROC_all[1]), batch_1_val, batch_2_val)

if __name__ == "__main__":

    main()
