# general import
import re
import shap
import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from pandas import to_pickle, read_pickle, DataFrame, Series

# machine learning model selection
class Machine_Learning_Method:
    LIGHTGBM = 'LightGBM'
    SVR = 'Support_Vector_Regression'
    LR = 'Logistic_Regression'

class WorkerResults(object):
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None,
                 y_pred_test=None, shap_values_all=None):
                 self.x_train = x_train
                 self.x_test = x_test
                 self.y_train = y_train
                 self.y_test = y_test
                 self.y_pred_test = y_pred_test
                 self.shap_values_all = shap_values_all

# build machine learning model
def ml_pipe(classifier, ml_method, params, seed_param, fe_hyper_param_keys, x_train, x_test, y_train, y_test, shap_importance):

    # do some clean up to avoid model warnings.
    if fe_hyper_param_keys is not None:
        params = {key: params[key] for key in params if key not in fe_hyper_param_keys}

    if ml_method == Machine_Learning_Method.LIGHTGBM:
        if classifier:
            model = LGBMClassifier(**seed_param) # defining our classifer based on params
        else:
            model = LGBMRegressor(**seed_param, verbose=-1)
        """ TO DO: use only lightGBM param here"""
        model.set_params(**params, verbose=-1)

    elif ml_method == Machine_Learning_Method.SVR:
        model = SVR(kernel=params['kernel'],
                    gamma=params['gamma'],
                    C=params['C'])

    elif ml_method == Machine_Learning_Method.LR:
        model = LogisticRegression(penalty=params['penalty'],
                                   solver=params['solver'],
                                   C=params['C'])


    model.fit(x_train, y_train.values.ravel())
    if classifier:
        y_pred_test = model.predict_proba(x_test)
    else:
        y_pred_test = model.predict(x_test)

    result = WorkerResults(x_train = None,
                           x_test = None,
                           y_train = None,
                           y_test = y_test,
                           y_pred_test = y_pred_test)

    if shap_importance:
        result.shap_values_all = shap.TreeExplainer(model).shap_values(x_train)
        result.x_train = x_train

    return result
