import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

def eval_nest(res):
    test_28 = [np.concatenate([res[o_s][nest][0][0].y_test<28 for nest in range(len(res[o_s]))]) for o_s in range(len(res))]
    test_32 = [np.concatenate([res[o_s][nest][0][0].y_test<32 for nest in range(len(res[o_s]))]) for o_s in range(len(res))]
    test_37 = [np.concatenate([res[o_s][nest][0][0].y_test<37 for nest in range(len(res[o_s]))]) for o_s in range(len(res))]
    pred = [np.concatenate([np.nan_to_num(-stats.zscore(res[o_s][nest][0][0].y_pred_test), nan=1)\
                                                    for nest in range(len(res[o_s]))]) for o_s in range(len(res))]
    return test_28, test_32, test_37, pred

def eval_nest_rs(res):
    test_28 = [np.concatenate([np.concatenate([list(res[0][j][i][0][0].y_test < 28),\
               list(res[1][j][i][0][0].y_test < 28)]) for i in range(len(res[0][j]))]) for j in range(len(res[0]))]
    test_32 = [np.concatenate([np.concatenate([list(res[0][j][i][0][0].y_test < 32),\
               list(res[1][j][i][0][0].y_test < 32)]) for i in range(len(res[0][j]))]) for j in range(len(res[0]))]
    test_37 = [np.concatenate([np.concatenate([list(res[0][j][i][0][0].y_test < 37),\
               list(res[1][j][i][0][0].y_test < 37)]) for i in range(len(res[0][j]))]) for j in range(len(res[0]))]
    pred = [np.concatenate([np.concatenate([list(np.nan_to_num(-stats.zscore(res[0][j][i][0][0].y_pred_test), nan=1)),\
            list(np.nan_to_num(-stats.zscore(res[1][j][i][0][0].y_pred_test), nan=1))]) for i in range(len(res[0][j]))]) for j in range(len(res[0]))]
    return test_28, test_32, test_37, pred

def eval_external(res, ptb = True):
    if ptb == True:
        test = res[0][0].y_test
    else:
        test = res[0][0].y_test < 37
    pred = np.nan_to_num(-stats.zscore(res[0][0].y_pred_test), nan=1)
    return test, pred

def eval_benchmark(res):
    y_test_all, y_pred_all = [], []
    auROC_all = []
    for n in res:

        y_test_outer, y_pred_outer = [], []
        auROC_outer = []
        for i in range(len(n)):
            y_test, y_pred = [], []
            for j in range(len(n[i])):
                y_test.append(list(n[i][j][0][0].y_test < 37))
                y_pred.append(np.nan_to_num(-stats.zscore(n[i][j][0][0].y_pred_test), nan=1))

            y_test_outer.append(np.concatenate(y_test))
            y_pred_outer.append(np.concatenate(y_pred))
            auROC_outer.append(roc_auc_score(np.concatenate(y_test), np.concatenate(y_pred)))
        y_test_all.append(y_test_outer)
        y_pred_all.append(y_pred_outer)
        auROC_all.append(auROC_outer)

    return y_test_all, y_pred_all, auROC_all

def eval_benchmark_cat(res_cat):
    y_test_all_cat, y_pred_all_cat = [], []
    auROC_all_cat = []
    for n in res_cat:

        y_test_outer, y_pred_outer = [], []
        auROC_outer = []
        for i in range(len(n[0])):
            y_test, y_pred = [], []
            for j in range(len(n[0][0])):
                y_test.append(np.concatenate([list(n[0][i][j][0][0].y_test < 37), list(n[1][i][j][0][0].y_test < 37)]))
                y_pred.append(np.concatenate([np.nan_to_num(-stats.zscore(n[0][i][j][0][0].y_pred_test), nan=1), \
                                             np.nan_to_num(-stats.zscore(n[1][i][j][0][0].y_pred_test), nan=1)]))

            y_test_outer.append(np.concatenate(y_test))
            y_pred_outer.append(np.concatenate(y_pred))
            auROC_outer.append(roc_auc_score(np.concatenate(y_test), np.concatenate(y_pred)))
        y_test_all_cat.append(y_test_outer)
        y_pred_all_cat.append(y_pred_outer)
        auROC_all_cat.append(auROC_outer)

    return y_test_all_cat, y_pred_all_cat, auROC_all_cat

def eval_race_val(res):
    test = res.y_test < 37
    pred = np.nan_to_num(-stats.zscore(res.y_pred_test), nan=1)
    auROC = roc_auc_score(test, pred)
    return test, pred, auROC

def eval_ns_to_s(res, AA_Samples, non_AA_Samples):
    res = np.concatenate(res)

    y_test_met6, y_pred_met6 = [], []
    y_test_met6_AA, y_pred_met6_AA = [], []
    y_test_met6_non_AA, y_pred_met6_non_AA = [], []

    auROC_met6 = []
    auROC_met6_AA = []
    auROC_met6_non_AA = []

    y_test, y_pred = [], []
    y_test_AA, y_pred_AA = [], []
    y_test_non_AA, y_pred_non_AA = [], []

    count = 0
    for i in range(50):
        count +=1

        tmp_df = pd.DataFrame(res[i][0][0].y_test)
        tmp_df["y_pred"] = res[i][0][0].y_pred_test
        tmp_AA = [i for i in tmp_df.index if i in AA_Samples]
        tmp_non_AA = [i for i in tmp_df.index if i in non_AA_Samples]
        tmp_df_AA = tmp_df.loc[tmp_AA]
        tmp_df_non_AA = tmp_df.loc[tmp_non_AA]

        y_test.append(list(res[i][0][0].y_test < 37))
        y_test_AA.append(list(tmp_df_AA["gw"] < 37))
        y_test_non_AA.append(list(tmp_df_non_AA["gw"] < 37))

        y_pred.append(np.nan_to_num(-stats.zscore(res[i][0][0].y_pred_test), nan=1))
        y_pred_AA.append(np.nan_to_num(-stats.zscore(tmp_df_AA["y_pred"]), nan=1))
        y_pred_non_AA.append(np.nan_to_num(-stats.zscore(tmp_df_non_AA["y_pred"]), nan=1))

        if count == 10:
            y_test_met6.append(np.concatenate(y_test))
            y_test_met6_AA.append(np.concatenate(y_test_AA))
            y_test_met6_non_AA.append(np.concatenate(y_test_non_AA))

            y_pred_met6.append(np.concatenate(y_pred))
            y_pred_met6_AA.append(np.concatenate(y_pred_AA))
            y_pred_met6_non_AA.append(np.concatenate(y_pred_non_AA))

            auROC_met6.append(roc_auc_score(np.concatenate(y_test), np.concatenate(y_pred)))
            auROC_met6_AA.append(roc_auc_score(np.concatenate(y_test_AA), np.concatenate(y_pred_AA)))
            auROC_met6_non_AA.append(roc_auc_score(np.concatenate(y_test_non_AA), np.concatenate(y_pred_non_AA)))

            y_test, y_pred = [], []
            y_test_AA, y_pred_AA = [], []
            y_test_non_AA, y_pred_non_AA = [], []
            count = 0

    return y_test_met6_AA, y_pred_met6_AA, y_test_met6_non_AA, y_pred_met6_non_AA, auROC_met6_AA, auROC_met6_non_AA
