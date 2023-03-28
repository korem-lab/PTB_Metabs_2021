import shap
import statistics
import numpy as np
import pandas as pd
from scipy import stats
from numpy import interp
from sklearn import metrics
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve


def main_auroc(test_list_all_37, pred_list_all, auroc_list_all_37, combination, age_bmi=True):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(2.7,2.7))

    # show or exclude combination prediction
    if age_bmi == True:
        if not combination:
            color_set = ['tab:orange','tab:green','tab:purple']
            label_set = ['Clinical', 'Microbiome', 'Metabolomics']
        else:
            color_set = ['tab:purple', 'tab:red']
            label_set = ['Metabolomics', 'Combination']
    else:
        color_set = ['tab:green','tab:purple']
        label_set = ['Microbiome', 'Metabolomics']

    count = 0
    for dataset in range(len(color_set)):
        for i in range(5):
            fpr, tpr, _ = roc_curve(test_list_all_37[dataset][i], pred_list_all[dataset][i], drop_intermediate=False)
            plt.plot(fpr, tpr, 'b', alpha=0.1, color=color_set[dataset])

            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

        auc_t = statistics.mean(auroc_list_all_37[dataset])
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        plt.plot(base_fpr, mean_tprs, 'b', lw=2, color=color_set[dataset], marker='',\
                 label=label_set[dataset]+' (auROC = %0.2f)' % auc_t)
        tprs = []
        base_fpr = np.linspace(0, 1, 101)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend(loc='lower right', fontsize=5)

    if not combination:
        plt.savefig('figurePanels/4A.pdf', dpi=500)
    else:
        plt.savefig('figurePanels/ext_data_8C.pdf', dpi=500)


def main_auprc(test_list_all_37, pred_list_all, auprc_list_all, combination, age_bmi=True):
    fprs = []
    base_tpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(2.7,2.7))

    plt.plot([0.34, 0.34], 'r--', lw=2, linestyle='--', label = 'Class balance (ratio = 0.34)')

    # show or exclude combination prediction
    if age_bmi == True:
        if not combination:
            color_set = ['tab:orange','tab:green','tab:purple']
            label_set = ['Clinical', 'Microbiome', 'Metabolomics']
        else:
            color_set = ['tab:purple', 'tab:red']
            label_set = ['Metabolomics', 'Combination']
    else:
        color_set = ['tab:green','tab:purple']
        label_set = ['Microbiome', 'Metabolomics']

    count = 0
    for dataset in range(len(color_set)):
        for i in range(5):
            fpr, tpr, _ = precision_recall_curve(test_list_all_37[dataset][i], pred_list_all[dataset][i])
            plt.plot(tpr, fpr, 'b', alpha=0.1, color=color_set[dataset])

            fpr = interp(base_tpr, tpr, fpr, period=1.1)
            fpr[0] = 0.0
            fprs.append(fpr)

        auc_t = statistics.mean(auprc_list_all[dataset])
        fprs = np.array(fprs)
        mean_fprs = fprs.mean(axis=0)
        std = fprs.std(axis=0)

        fprs_upper = np.minimum(mean_fprs + std, 1)
        fprs_lower = mean_fprs - std

        plt.plot(base_tpr, mean_fprs, 'b', lw=2, color=color_set[dataset], marker='', \
                 label=label_set[dataset]+' (auPR = %0.2f)' % auc_t)

        fprs = []
        base_tpr = np.linspace(0, 1, 101)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend(loc='lower right', fontsize=5, handlelength=4.1)
    if not combination:
        plt.savefig('figurePanels/4B.pdf', dpi=500)
    else:
        plt.savefig('figurePanels/ext_data_8D.pdf', dpi=500)


def gw_thresh(test_list_all_AA, pred_list_all_AA, auroc_list_all_AA, microbiome):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(2.7,2.7))

    if microbiome:
        color_set = ['palegreen','limegreen','darkgreen']
    else:
        color_set = ['violet','darkviolet','rebeccapurple']
    label_set = ['28 weeks', '32 weeks', '37 weeks']

    for gw_idx in range(3):
        for i in range(5):
            if microbiome:
                fpr, tpr, _ = roc_curve(test_list_all_AA[gw_idx][0][i], pred_list_all_AA[0][i], drop_intermediate=False)
            else:
                fpr, tpr, _ = roc_curve(test_list_all_AA[gw_idx][1][i], pred_list_all_AA[1][i], drop_intermediate=False)

            plt.plot(fpr, tpr, 'b', alpha=0.1, color=color_set[gw_idx])

            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

        if microbiome:
            auc_t = statistics.mean(auroc_list_all_AA[gw_idx][0])
        else:
            auc_t = statistics.mean(auroc_list_all_AA[gw_idx][1])

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        plt.plot(base_fpr, mean_tprs, 'b', lw=2, color=color_set[gw_idx], marker='',
                 label=label_set[gw_idx]+' (auROC = %0.2f)' % auc_t)

        tprs = []
        base_fpr = np.linspace(0, 1, 101)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend(loc='lower right', fontsize=5)
    if microbiome:
        plt.savefig('figurePanels/ext_data_8G.pdf', dpi=500)
    else:
        plt.savefig('figurePanels/ext_data_8F.pdf', dpi=500)


def external_auroc(t1, p1, t2, p2):

    fpr_1, tpr_1, thresholds_1 = metrics.roc_curve(t1, p1, drop_intermediate=False)
    auc_1 = roc_auc_score(t1, p1)
    fpr_2, tpr_2, thresholds_2 = metrics.roc_curve(t2, p2, drop_intermediate=False)
    auc_2 = roc_auc_score(t2, p2)

    c=['olive', 'Teal']
    plt.figure(figsize=(2.7,2.7))
    lw = 2

    plt.plot(fpr_1, tpr_1, color=c[0], lw=lw, linestyle = '-', label='Ghartey 2017 (auROC = %0.2f)' % auc_1)
    plt.plot(fpr_2, tpr_2, color=c[1], lw=lw, linestyle = '-', label='Ghartey 2015 (auROC = %0.2f)' % auc_2)
    plt.plot([0, 1], [0, 1], 'r--', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", fontsize=5)
    plt.savefig('figurePanels/4C.pdf', dpi=500)


def external_auprc(t1, p1, t2, p2):
    fpr_1, tpr_1, thresholds_1 = precision_recall_curve(t1, p1)
    auc_1 = average_precision_score(t1, p1)
    c=['olive']
    plt.figure(figsize=(2.7,2.7))
    lw = 2
    plt.plot([0.4, 0.4], 'r--', lw=lw, linestyle='--', label = 'Class balance (ratio = 0.40)')
    plt.plot(tpr_1, fpr_1, color=c[0], lw=lw, linestyle = '-', label='Ghartey 2017 (auPR = %0.2f)' % auc_1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", fontsize=5, handlelength=4.1)
    plt.savefig('figurePanels/ext_data_8H.pdf', dpi=500)

    fpr_2, tpr_2, thresholds_2 = precision_recall_curve(t2, p2)
    auc_2 = average_precision_score(t2, p2)
    c=['teal']
    plt.figure(figsize=(2.7,2.7))
    lw = 2
    plt.plot([0.5, 0.5], "r--", lw=lw, linestyle='--', label = 'Class balance (ratio = 0.50)')
    plt.plot(tpr_2, fpr_2, color=c[0], lw=lw, linestyle = '-', label='Ghartey 2015 (auPR = %0.2f)' % auc_2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", fontsize=5, handlelength=4.1)
    plt.savefig('figurePanels/ext_data_8I.pdf', dpi=500)


def micro_shap(res_1, res_2, otu):
    x_train_AA = pd.DataFrame(res_1[0][0].x_train)
    x_train_non_AA = pd.DataFrame(res_2[0][0].x_train)
    af_AA = pd.DataFrame(data=res_1[0][0].shap_values_all, columns=x_train_AA.columns, index = x_train_AA.index)
    af_non_AA = pd.DataFrame(data=res_2[0][0].shap_values_all, columns=x_train_non_AA.columns, index = x_train_non_AA.index)
    af = af_AA.append(af_non_AA , sort=False).fillna(0)
    x_train = x_train_AA.append(x_train_non_AA, sort=False)

    af2 = af.copy()
    if 'alpha' in af2.columns:
        temp = af['alpha']
        af = af.drop('alpha', axis=1)

    af = af.rename(columns = lambda r: '%s (%s)' % (r, (['%s: %s' % \
                  (hot[0], otu.loc[r, hot].replace('_',' ').replace('\"', '').replace('\\', '')) \
                   for hot in ('Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum') \
                   if pd.notnull(otu.loc[r, hot])] + ['Unknown'])[0]))

    if 'alpha' in af2.columns:
        af['alpha'] = temp

    x_train.columns = af.columns

    plt.figure()
    shap.summary_plot(af.values, x_train,
                      plot_type="dot",
                      max_display=10,
                      show=False,
                      color_bar=True,
                      color_bar_label = '')

    plt.gcf()
    af.to_csv("shap_tables/shap_table_microbiome.csv")
    plt.savefig('figurePanels/ext_data_8J_yticks.pdf', dpi = 500, figsize=(2.273,2.2322), bbox_inches = 'tight')

    plt.yticks([], [])
    plt.xlabel("")
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xticks([-2, -1, 0, 1])
    cax = fig.axes[-1]
    cax.tick_params(labeltop="", labelbottom="", labelsize=0, tick1On=False)

    plt.savefig('figurePanels/ext_data_8J.pdf', dpi=500, figsize=(2.273, 2.2322), bbox_inches='tight')


def metab_shap(res_1, res_2, id):
    x_train_AA = pd.DataFrame(res_1[0][0].x_train)
    x_train_non_AA = pd.DataFrame(res_2[0][0].x_train)
    af_AA = pd.DataFrame(data=res_1[0][0].shap_values_all, columns=x_train_AA.columns, index = x_train_AA.index)
    af_non_AA = pd.DataFrame(data=res_2[0][0].shap_values_all, columns=x_train_non_AA.columns, index = x_train_non_AA.index)
    af = af_AA.append(af_non_AA , sort=False).fillna(0)
    x_train = x_train_AA.append(x_train_non_AA, sort=False)
    #shap.initjs()

    del af['alpha']
    del x_train['alpha']
    af = af.rename(columns = id)
    x_train.columns = af.columns

    plt.figure()
    shap.summary_plot(af.values, x_train,
                      plot_type="dot",
                      max_display=10,
                      show=False,
                      color_bar=True,
                      color_bar_label = '')

    plt.gcf()
    af.to_csv("shap_tables/shap_table_metabolomics.csv")
    plt.savefig('figurePanels/4D_yticks.pdf', dpi =500, figsize=(2.273,2.2322), bbox_inches = 'tight')

    plt.yticks([],[])
    plt.xlabel("")
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xticks([-2, -1, 0, 1])
    cax = fig.axes[-1]
    cax.tick_params(labeltop="", labelbottom="", labelsize=0, tick1On=False)

    plt.savefig('figurePanels/4D.pdf', dpi=500, figsize=(2.273, 2.2322), bbox_inches='tight')


def combo_shap(res_1, res_2, id):
    x_train_AA = pd.DataFrame(res_1[0][0].x_train)
    x_train_non_AA = pd.DataFrame(res_2[0][0].x_train)
    af_AA = pd.DataFrame(data=res_1[0][0].shap_values_all, columns=x_train_AA.columns, index = x_train_AA.index)
    af_non_AA = pd.DataFrame(data=res_2[0][0].shap_values_all, columns=x_train_non_AA.columns, index = x_train_non_AA.index)
    af = af_AA.append(af_non_AA , sort=False).fillna(0)
    x_train = x_train_AA.append(x_train_non_AA, sort=False)

    id = {str(key) + "_metab": val for key, val in id.items()}
    af = af.rename(columns = id)
    x_train.columns = af.columns

    plt.figure()
    shap.summary_plot(af.values, x_train,
                      plot_type="dot",
                      max_display=10,
                      show=False,
                      color_bar=True,
                      color_bar_label = '')

    plt.gcf()
    af.to_csv("shap_tables/shap_table_combination.csv")
    plt.savefig('figurePanels/ext_data_8E_yticks.pdf', dpi =500, figsize=(2.273,2.2322), bbox_inches = 'tight')

    plt.yticks([], [])
    plt.xlabel("")
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xticks([-2, -1, 0, 1])
    cax = fig.axes[-1]
    cax.tick_params(labeltop="", labelbottom="", labelsize=0, tick1On=False)

    plt.savefig('figurePanels/ext_data_8E.pdf', dpi=500, figsize=(2.273, 2.2322), bbox_inches='tight')


def SV_LR_comparison(y_test_all_cat, y_pred_all_cat, auROC_all_cat):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(2.7,2.7))

    color_set = ['tab:cyan', 'tab:pink', 'tab:gray', 'xkcd:light lavender']
    label_set = ['LightGBM', 'Support Vector Classification', 'Logistic Regression', 'Elastic Net']

    count = 0
    for dataset in range(4):
        for i in range(5):
            fpr, tpr, _ = roc_curve(y_test_all_cat[dataset][i], y_pred_all_cat[dataset][i], drop_intermediate=False)
            plt.plot(fpr, tpr, 'b', alpha=0.1, color=color_set[dataset])

            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

        auc_t = statistics.mean(auROC_all_cat[dataset])
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        plt.plot(base_fpr, mean_tprs, 'b', lw=2, color=color_set[dataset], marker='',\
                 label=label_set[dataset]+' (auROC = %0.2f)' % auc_t)
        tprs = []
        base_fpr = np.linspace(0, 1, 101)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend(loc='lower right', fontsize=5)
    plt.savefig('figurePanels/ext_data_8A.pdf', dpi=500)


def race_strat_compare_to_non_strat(y_test_all_cat, y_pred_all_cat, auROC_all_cat, y_test_all, y_pred_all, auROC_all):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(2.7,2.7))

    color_set = ['tab:orange', 'tab:gray']
    label_set = ['Race-stratified', 'Non-race-stratified']

    count = 0
    for dataset in range(2):
        for i in range(5):
            if dataset == 0:
                fpr, tpr, _ = roc_curve(y_test_all_cat[0][i], y_pred_all_cat[0][i], drop_intermediate=False)
            else:
                fpr, tpr, _ = roc_curve(y_test_all[6][i], y_pred_all[6][i], drop_intermediate=False)

            plt.plot(fpr, tpr, 'b', alpha=0.1, color=color_set[dataset])

            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

        if dataset == 0:
            auc_t = statistics.mean(auROC_all_cat[0])
        else:
            auc_t = statistics.mean(auROC_all[6])
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        plt.plot(base_fpr, mean_tprs, 'b', lw=2, color=color_set[dataset], marker='',\
                 label=label_set[dataset]+' (auROC = %0.2f)' % auc_t)
        tprs = []
        base_fpr = np.linspace(0, 1, 101)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend(loc='lower right', fontsize=5)
    plt.savefig('figurePanels/race_strat_comparison.pdf', dpi=500)


def model_performance_by_race(y_test_r, y_pred_r, auROC_r, y_test_nr, y_pred_nr, auROC_nr):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(2.7,2.7))

    color_set = ['tab:red', 'tab:blue', 'rosybrown', 'lightsteelblue']
    label_set = ['AA CV', 'non-AA CV', 'Non-race-stratified CV on AA', 'Non-race-stratified CV on non-AA']

    count = 0
    for dataset in range(4):
        for i in range(5):
            fpr, tpr, _ = roc_curve(y_test_r[dataset][i], y_pred_r[dataset][i], drop_intermediate=False)
            plt.plot(fpr, tpr, 'b', alpha=0.1, color=color_set[dataset])

            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

            auc_t = statistics.mean(auROC_r[dataset])

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        plt.plot(base_fpr, mean_tprs, 'b', lw=2, color=color_set[dataset], marker='',\
                 label=label_set[dataset]+' (auROC = %0.2f)' % auc_t)
        tprs = []
        base_fpr = np.linspace(0, 1, 101)

    fpr_1, tpr_1, thresholds_1 = metrics.roc_curve(y_test_nr[0], y_pred_nr[0], drop_intermediate=False)
    auc_1 = auROC_nr[0]
    fpr_2, tpr_2, thresholds_2 = metrics.roc_curve(y_test_nr[1], y_pred_nr[1], drop_intermediate=False)
    auc_2 = auROC_nr[1]

    c=['orange', 'darkturquoise']
    lw = 2
    plt.plot(fpr_1, tpr_1, color=c[0], lw=lw, linestyle = '-', label='AA validate on non-AA (auROC = %0.2f)' % auc_1)
    plt.plot(fpr_2, tpr_2, color=c[1], lw=lw, linestyle = '-', label='non-AA validate on AA (auROC = %0.2f)' % auc_2)


    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend(loc='lower right', fontsize=5)
    plt.savefig('figurePanels/ext_data_8B_with_legend.pdf', dpi=500)

    plt.gca().get_legend().remove()
    plt.savefig('figurePanels/ext_data_8B.pdf', dpi=500)


def batch_auroc(rs_acc, batch_1, batch_2, batch_1_val, batch_2_val):
    plt.figure(figsize=(2.7,2.7))
    mu, std = norm.fit(rs_acc)

    # Plot the histogram.
    plt.hist(rs_acc, bins=25, density=True, alpha=0.6, rwidth=0.85, color='tab:gray')

    ## print the p values
    print(rs_acc)
    print(batch_1_val)
    print(batch_2_val)
    print(batch_1)
    print(batch_2)

    # Plot the kde.
    kde = stats.gaussian_kde(rs_acc)
    xx = np.linspace(0.55, 0.8)
    plt.plot(xx, kde(xx), 'k')

    plt.axvline(x=batch_1_val, color='tab:purple', label = 'batch 1 validate on batch 2, (auROC = %0.2f)' % batch_1_val)
    plt.axvline(x=batch_2_val, color='tab:brown', label = 'batch 2 validate on batch 1, (auROC = %0.2f)' % batch_2_val)
    plt.axvline(x=batch_1, color='tab:orange', label = 'batch 1 CV, (auROC = %0.2f)' % batch_1)
    plt.axvline(x=batch_2, color='tab:pink', label = 'batch 2 CV, (auROC = %0.2f)' % batch_2)

    plt.legend(loc='lower right', fontsize=5)
    plt.bbox_inches='tight'
    plt.savefig('figurePanels/ext_data_2E_with_legend.pdf', dpi=500)

    plt.gca().get_legend().remove()
    plt.savefig('figurePanels/ext_data_2E.pdf', dpi=500)


def compare_aucs(au1, au2):
    """
    au1- list of aucs
    au2 - other list of aucs
    """
    au1 = np.array(au1)
    au2 = np.array(au2)

    m1, m2 = au1.mean(), au2.mean()
    r = stats.pearsonr(au1, au2)[0]
    print(m1, m2, r)
    s1, s2 = np.std(au1), np.std(au2)

    z = (m1 - m2) / np.sqrt(np.power(s1, 2) + np.power(s2, 2) - r * s1 * s2)
    p = 2*stats.norm.sf(abs(z))
    print("p-value: " + str(p))
    print("------------------")
    return (p)