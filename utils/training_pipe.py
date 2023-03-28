# general import
from utils.feature_engineering import fe_pipe
from utils.machine_learning import ml_pipe


# study design for different run purposes
class Study_Design:
    NORMAL_OPTIMIZATION = 'Normal_Optimization' # normal optimization run
    NESTED_OPTIMIZATION = 'Nested_Optimization' # nested optimization run
    RANDOM_SAMPLING_OPTIMIZATION = 'Random_Sampling_Optimization' # randomly draw some samples from the data and optimize
    CHECK_SHAP_IMPORTANCE = 'Check_Shap_Importance' # check feature importance using shap
    NESTED_VALIDATION = 'Nested_Validation' # nested validation setting (for nested prediction)
    NESTED_P_VALUE = 'Nessted_p_value' # generate nested p values
    EXTERNAL_VALIDATION = 'External_Validation' # external validation setting (to test other cohort)



# Performs the entire pipeline for one CV fold for given set of params
def pipe(X, md, params_list, cv, classifier, ml_method, study_design, train_samples_to_remove,
         ev_data = None, ev_md = None):

    # set target for y
    if classifier:
        y = md.loc[:, "sPTB"]
    else:
        y = md.loc[:, "gw"]

    # automatically detect if it is combination model
    try:
        combination = X.columns.str.contains('age_clin').any()
    except:
        combination = False

    # Seeding for lightGBM
    seed = {'n_jobs': 1,
            'subsample_freq': 5,
            'random_state': 420,
            'bagging_seed': 420,
            'feature_fraction_seed': 420,
            'data_random_seed': 420,
            'extra_seed': 420}

    # Separate model parameters from feature engineering parameters
    fe_hyper_param_keys = ['do_imputate','imputate_method','do_standardize','std_method','transformation','do_pca','pca_method',
                           'nPC','do_feature_selection','fs_threshold','fs_method','do_feature_filtering','ff_threshold','do_shap_selection',
                           'shap_threshold','do_info_gain','ig_threshold','do_diversity_adding','diversity_metric','do_ycap',
                           'do_remove_from_train', 'do_total_reads_adding']

    # minor constant addition for log
    log_eps = 1e-6

    # parameter space set up
    if type(params_list) == dict:
        params_list = [params_list]
    for i in params_list:
        try:
            i['min_split_gain'] = i['min_gain_to_split']
            del i['min_gain_to_split']
        except:
            pass

    # external data
    if study_design!=Study_Design.EXTERNAL_VALIDATION:
        ev_test, ev_y = None, None
    else:
        try:
            ev_test, ev_y = ev_data, ev_md['gw']
        except:
            ev_test, ev_y = ev_data, ev_md['ptb']


    # feature engineering pipeline
    result_list = []
    for params in params_list:
        result = []
        for _, (train_index, test_index) in cv.items():
            # feature engineering pipe
            x_train, y_train, x_test, y_test = fe_pipe(do_imputate=params['do_imputate'],
                                                      imputate_method=params['imputate_method'],
                                                      do_standardize=params['do_standardize'],
                                                      std_method=params['std_method'],
                                                      transformation=params['transformation'],
                                                      do_pca=params['do_pca'],
                                                      pca_method=params['pca_method'],
                                                      nPC=params['nPC'],
                                                      do_feature_selection=params['do_feature_selection'],
                                                      fs_threshold=params['fs_threshold'],
                                                      fs_method=params['fs_method'],
                                                      do_feature_filtering=params['do_feature_filtering'],
                                                      ff_threshold=params['ff_threshold'],
                                                      do_shap_selection=params['do_shap_selection'],
                                                      shap_threshold=params['shap_threshold'],
                                                      do_info_gain=params['do_info_gain'],
                                                      ig_threshold=params['ig_threshold'],
                                                      do_diversity_adding=params['do_diversity_adding'],
                                                      diversity_metric=params['diversity_metric'],
                                                      do_total_reads_adding=params['do_total_reads_adding'],
                                                      do_ycap=params['do_ycap'],
                                                      do_remove_from_train=params['do_remove_from_train'],
                                                      train_samples_to_remove=train_samples_to_remove,
                                                      fe_hyper_param_keys=fe_hyper_param_keys,
                                                      train_index=train_index,
                                                      test_index=test_index,
                                                      x=X,
                                                      y=y,
                                                      metadata=md,
                                                      external_validation=(study_design==Study_Design.EXTERNAL_VALIDATION),
                                                      ev_test=ev_test,
                                                      ev_y=ev_y,
                                                      shap_importance=(study_design==Study_Design.CHECK_SHAP_IMPORTANCE),
                                                      combination=combination,
                                                      params=params,
                                                      classifier=classifier,
                                                      ml_method=ml_method,
                                                      seed=seed,
                                                      log_eps=log_eps)

            # model training pipeline
            result.append(ml_pipe(classifier,
                                  ml_method,
                                  params,
                                  seed,
                                  fe_hyper_param_keys,
                                  x_train,
                                  x_test,
                                  y_train,
                                  y_test,
                                  study_design==Study_Design.CHECK_SHAP_IMPORTANCE))

        result_list.append(result)

    return result_list
