import os
import inspect
import re
import warnings
import shap
import pickle
import statistics
import pandas as pd
from pandas import Series, DataFrame, concat, to_pickle
import numpy as np
from numpy import interp
from scipy import stats
from sklearn import metrics
from sklearn.metrics import auc, roc_auc_score, roc_curve, mean_squared_error, average_precision_score, precision_recall_curve, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sn
from operator import itemgetter
import shap

def figure_S8ABC():

	# Meta
	res_1_A = pd.read_pickle("final_models/meta.pkl") #dummy
	res_2_A = pd.read_pickle("final_models/meta.pkl") #dummy
	res_A = pd.read_pickle("final_models/meta.pkl")

	# OTU
	res_1_B = pd.read_pickle("final_models/micro_AA.pkl")
	res_2_B = pd.read_pickle("final_models/micro_10_non_AA.pkl")
	res_B = pd.read_pickle("final_models/meta.pkl") # dummy

	# Bolo
	res_1_C = pd.read_pickle("final_models/bolo_10_AA.pkl")
	res_2_C = pd.read_pickle("final_models/bolo_non_AA.pkl")
	res_C = pd.read_pickle("final_models/meta.pkl") # dummy

	# Combo
	res_1_D = pd.read_pickle("final_models/combo_10_AA.pkl")
	res_2_D = pd.read_pickle("final_models/combo_non_AA.pkl")
	res_D = pd.read_pickle("final_models/meta.pkl") #dummy

	res = []
	res.append([[res_1_A, res_2_A], [res_1_B, res_2_B], [res_1_C, res_2_C], [res_1_D, res_2_D]])
	res.append([[res_A], [res_B], [res_C], [res_D]])

	gw_same_predictor = True
	# order to be filled : stratified, non stratified
	# nested order: clinical, Microbiome, Metabolomics, Combination
	# next nested order: AA, non-AA
	best_auroc_37_idx_strat = [[[0, 0], [0, 1], [7, 0], [8, 0]], [[0], [0], [0], [0]]]
	best_auprc_37_idx_strat = [[[0, 0], [0, 1], [7, 0], [8, 0]], [[0], [0], [0], [0]]]

	gw_best_y_test_all_auroc=[]
	gw_best_y_pred_all_auroc=[]
	gw_best_y_test_all_auprc=[]
	gw_best_y_pred_all_auprc=[]
	gw_best_auroc_all=[]
	gw_best_auprc_all=[]

	ns_gw_best_y_test_all_auroc=[]
	ns_gw_best_y_pred_all_auroc=[]
	ns_gw_best_y_test_all_auprc=[]
	ns_gw_best_y_pred_all_auprc=[]
	ns_gw_best_auroc_all=[]
	ns_gw_best_auprc_all=[]

	stratify_or_not = ["stratified", "non-stratified"]
	for i in range(2):
	    gw_list = [28, 32, 37]
	    for gw in gw_list:
	        best_y_test_all_auroc=[]
	        best_y_pred_all_auroc=[]
	        best_y_test_all_auprc=[]
	        best_y_pred_all_auprc=[]
	        best_auroc_all=[]
	        best_auprc_all=[]

	        data_type_list = ["Clinical", "Microbiome", "Metabolomics", "Combination"]
	        for dataset_idx in range(len(res[i])):
	            # do metadata, microbiome, metabolomics at the same time.

	            best_y_test_dataset_auroc=[]
	            best_y_pred_dataset_auroc=[]
	            best_y_test_dataset_auprc=[]
	            best_y_pred_dataset_auprc=[]
	            best_auroc_dataset=[]
	            best_auprc_dataset=[]

	            strat_kind = ["AA", "non_AA"]
	            for strat in range(len(res[i][dataset_idx])):

	                # calculate auroc and auprc for one robust run
	                y_test_strat=[]
	                y_pred_strat=[]
	                auroc_strat=[]
	                auprc_strat=[]
	                r2_strat=[]
	                for iter in res[i][dataset_idx][strat]:
	                    y_test=[]
	                    y_pred=[]
	                    y_test_r2=[]
	                    y_pred_r2=[]
	                    auroc_ef=[]
	                    auprc_ef=[]
	                    r2_ef=[]

	                    count = 0
	                    y_test_r = []
	                    y_pred_r = []
	                    y_test_r_r2 = []
	                    y_pred_r_r2 = []
	                    for fold in iter[2]:
	                        count += 1
	                        y_test.append(fold[1].y_test < gw)
	                        y_pred.append(np.nan_to_num(-stats.zscore(fold[1].y_pred_test), nan=1))
	                        y_test_r2.append(fold[1].y_test)
	                        y_pred_r2.append(fold[1].y_pred_test)

	                        if count == 10:
	                            y_test_r.append(np.concatenate(y_test))
	                            y_pred_r.append(np.concatenate(y_pred))
	                            y_test_r_r2.append(np.concatenate(y_test_r2))
	                            y_pred_r_r2.append(np.concatenate(y_pred_r2))

	                            auroc_ef.append(roc_auc_score(np.concatenate(y_test), np.concatenate(y_pred)))
	                            auprc_ef.append(average_precision_score(np.concatenate(y_test), np.concatenate(y_pred)))
	                            r2_ef.append(r2_score(np.concatenate(y_test_r2), np.concatenate(y_pred_r2)))

	                            count = 0
	                            y_test = []
	                            y_pred = []

	                    y_test_strat.append(y_test_r)
	                    y_pred_strat.append(y_pred_r)
	                    auroc_strat.append(statistics.mean(auroc_ef))
	                    auprc_strat.append(statistics.mean(auprc_ef))
	                    r2_strat.append(statistics.mean(r2_ef))

	                # get the best auroc and best auprc and their indeces, record the test and pred
	                df_strat = pd.DataFrame()
	                df_strat["auroc"] = auroc_strat
	                df_strat["auprc"] = auprc_strat
	                df_strat["r2"] = r2_strat

	                if gw_same_predictor:
	                    # manually enter the same predictor for 37 gw weeks
	                    best_auroc_strat = df_strat["auroc"][best_auroc_37_idx_strat[i][dataset_idx][strat]]
	                    best_auprc_strat = df_strat["auprc"][best_auprc_37_idx_strat[i][dataset_idx][strat]]
	                    y_test_dataset_auroc=y_test_strat[best_auroc_37_idx_strat[i][dataset_idx][strat]]
	                    y_pred_dataset_auroc=y_pred_strat[best_auroc_37_idx_strat[i][dataset_idx][strat]]
	                    y_test_dataset_auprc=y_test_strat[best_auprc_37_idx_strat[i][dataset_idx][strat]]
	                    y_pred_dataset_auprc=y_pred_strat[best_auprc_37_idx_strat[i][dataset_idx][strat]]

	                else:
	                    df_strat = df_strat.sort_values("auroc", ascending = False)
	                    best_auroc_strat_idx = df_strat.index.to_list()[0]
	                    best_auroc_strat = df_strat["auroc"][best_auroc_strat_idx]

	                    y_test_dataset_auroc=y_test_strat[best_auroc_strat_idx]
	                    y_pred_dataset_auroc=y_pred_strat[best_auroc_strat_idx]

	                    df_strat = df_strat.sort_values("auprc", ascending = False)
	                    best_auprc_strat_idx = df_strat.index.to_list()[0]
	                    best_auprc_strat = df_strat["auprc"][best_auprc_strat_idx]

	                    y_test_dataset_auprc=y_test_strat[best_auprc_strat_idx]
	                    y_pred_dataset_auprc=y_pred_strat[best_auprc_strat_idx]

	                best_y_test_dataset_auroc.append(y_test_dataset_auroc)
	                best_y_pred_dataset_auroc.append(y_pred_dataset_auroc)
	                best_y_test_dataset_auprc.append(y_test_dataset_auprc)
	                best_y_pred_dataset_auprc.append(y_pred_dataset_auprc)
	                best_auroc_dataset.append(best_auroc_strat)
	                best_auprc_dataset.append(best_auprc_strat)

	            best_y_test_all_auroc.append(best_y_test_dataset_auroc)
	            best_y_pred_all_auroc.append(best_y_pred_dataset_auroc)
	            best_y_test_all_auprc.append(best_y_test_dataset_auprc)
	            best_y_pred_all_auprc.append(best_y_pred_dataset_auprc)
	            best_auroc_all.append(best_auroc_dataset)
	            best_auprc_all.append(best_auprc_dataset)

	        if i == 0:
	            gw_best_y_test_all_auroc.append(best_y_test_all_auroc)
	            gw_best_y_pred_all_auroc.append(best_y_pred_all_auroc)
	            gw_best_y_test_all_auprc.append(best_y_test_all_auprc)
	            gw_best_y_pred_all_auprc.append(best_y_pred_all_auprc)
	            gw_best_auroc_all.append(best_auroc_all)
	            gw_best_auprc_all.append(best_auprc_all)
	        else:
	            ns_gw_best_y_test_all_auroc.append(best_y_test_all_auroc)
	            ns_gw_best_y_pred_all_auroc.append(best_y_pred_all_auroc)
	            ns_gw_best_y_test_all_auprc.append(best_y_test_all_auprc)
	            ns_gw_best_y_pred_all_auprc.append(best_y_pred_all_auprc)
	            ns_gw_best_auroc_all.append(best_auroc_all)
	            ns_gw_best_auprc_all.append(best_auprc_all)

	test_list_all = []
	pred_list_all = []
	auroc_list_all = []
	for i in range(len(gw_best_y_test_all_auroc[2])): # per dataset

	    test_list = []
	    pred_list = []
	    auroc_list = []
	    for j in range(len(gw_best_y_test_all_auroc[2][0][0])): # per cv
	        if i == 0: # non_stratified, clinical data
	            test = ns_gw_best_y_test_all_auroc[2][i][0][j]
	            pred = ns_gw_best_y_pred_all_auroc[2][i][0][j]
	            test_list.append(test)
	            pred_list.append(pred)
	            auroc_list.append(roc_auc_score(test, pred))
	        else:  # stratified, other data
	            test = np.concatenate([gw_best_y_test_all_auroc[2][i][0][j], gw_best_y_test_all_auroc[2][i][1][j]])
	            pred = np.concatenate([gw_best_y_pred_all_auroc[2][i][0][j], gw_best_y_pred_all_auroc[2][i][1][j]])
	            test_list.append(test)
	            pred_list.append(pred)
	            auroc_list.append(roc_auc_score(test, pred))

	    test_list_all.append(test_list)
	    pred_list_all.append(pred_list)
	    auroc_list_all.append(auroc_list)

	################### S8A #############################

	tprs = []
	base_fpr = np.linspace(0, 1, 101)
	plt.figure(figsize=(2.7,2.7))

	color_set = ['tab:orange','tab:green','tab:purple', 'tab:red']
	label_set = ['Clinical', 'Microbiome', 'Metabolomics', 'Combination']

	count = 0
	for psuedo_dataset in range(2):
	    dataset = psuedo_dataset + 2
	    for i in range(5):
	        fpr, tpr, _ = roc_curve(test_list_all[dataset][i], pred_list_all[dataset][i], drop_intermediate=False)
	        plt.plot(fpr, tpr, 'b', alpha=0.1, color=color_set[dataset])

	        tpr = interp(base_fpr, fpr, tpr)
	        tpr[0] = 0.0
	        tprs.append(tpr)

	    auc_t = statistics.mean(auroc_list_all[dataset])
	    tprs = np.array(tprs)
	    mean_tprs = tprs.mean(axis=0)
	    std = tprs.std(axis=0)

	    tprs_upper = np.minimum(mean_tprs + std, 1)
	    tprs_lower = mean_tprs - std

	    plt.plot(base_fpr, mean_tprs, 'b', lw=2, color=color_set[dataset], marker='', label=label_set[dataset]+' (auROC = %0.2f)' % auc_t)
	    tprs = []
	    base_fpr = np.linspace(0, 1, 101)
	    count +=1
	    if count == 4:
	        break

	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([-0.01, 1.01])
	plt.ylim([-0.01, 1.01])
	# plt.ylabel('True Positive Rate', fontsize=5)
	# plt.xlabel('False Positive Rate', fontsize=5)
	#plt.title("A. sPTB prediction accuracy (auROC)", weight='bold', fontsize=9)
	plt.legend(loc='lower right', fontsize=5)
	plt.savefig('figurePanels/S8A.png', dpi=800)
	plt.savefig('figurePanels/S8A.pdf', dpi=800)
	plt.close()

	################### S8B #############################

	test_list_all = []
	pred_list_all = []
	auprc_list_all = []
	for i in range(len(gw_best_y_test_all_auprc[2])): # per dataset
	    test_list = []
	    pred_list = []
	    auprc_list = []
	    for j in range(len(gw_best_y_test_all_auprc[2][0][0])): # per cv
	        if i == 0:# or i == 2:
	            test = ns_gw_best_y_test_all_auprc[2][i][0][j]
	            pred = ns_gw_best_y_pred_all_auprc[2][i][0][j]
	            test_list.append(test)
	            pred_list.append(pred)
	            auprc_list.append(average_precision_score(test, pred))
	        else:
	            test = np.concatenate([gw_best_y_test_all_auprc[2][i][0][j], gw_best_y_test_all_auprc[2][i][1][j]])
	            pred = np.concatenate([gw_best_y_pred_all_auprc[2][i][0][j], gw_best_y_pred_all_auprc[2][i][1][j]])
	            test_list.append(test)
	            pred_list.append(pred)
	            auprc_list.append(average_precision_score(test, pred))

	    test_list_all.append(test_list)
	    pred_list_all.append(pred_list)
	    auprc_list_all.append(auprc_list)

	fprs = []
	base_tpr = np.linspace(0, 1, 101)
	plt.figure(figsize=(2.7,2.7))
	plt.plot([0.34, 0.34], 'r--', lw=2, linestyle='--', label = 'Class balance (ratio = 0.34)')

	color_set = ['tab:orange','tab:green','tab:purple', 'tab:red']
	label_set = ['Clinical', 'Microbiome', 'Metabolomics', 'Combination']
	count = 0
	for psuedo_dataset in range(2):
	    dataset = psuedo_dataset + 2
	    for i in range(5):
	        fpr, tpr, _ = precision_recall_curve(test_list_all[dataset][i], pred_list_all[dataset][i])
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

	    plt.plot(base_tpr, mean_fprs, 'b', lw=2, color=color_set[dataset], marker='', label=label_set[dataset]+' (auPR = %0.2f)' % auc_t)

	    fprs = []
	    base_tpr = np.linspace(0, 1, 101)
	    count += 1
	    if count == 4:
	        break

	plt.xlim([-0.01, 1.01])
	plt.ylim([-0.01, 1.01])
	# plt.ylabel('True Positive Rate', fontsize=5)
	# plt.xlabel('False Positive Rate', fontsize=5)
	#plt.title("A. sPTB prediction accuracy (auROC)", weight='bold', fontsize=9)
	plt.legend(loc='lower right', fontsize=5, handlelength=4.1)
	plt.savefig('figurePanels/S8B.png', dpi=800)
	plt.savefig('figurePanels/S8B.pdf', dpi=800)
	plt.close()

	################### S8C #############################
	stratify = stratify = [True, True, True]
	test_list_gw = []
	pred_list_gw = []
	auroc_list_gw = []
	for i in range(len(gw_list)): # per dataset
	    test_list = []
	    pred_list = []
	    auroc_list = []
	    if stratify[i]:
	        for j in range(len(gw_best_y_test_all_auroc[2][0][0])): # per cv
	            test = np.concatenate([gw_best_y_test_all_auroc[i][2][0][j], gw_best_y_test_all_auroc[i][2][1][j]])
	            pred = np.concatenate([gw_best_y_pred_all_auroc[i][2][0][j], gw_best_y_pred_all_auroc[i][2][1][j]])
	            test_list.append(test)
	            pred_list.append(pred)
	            auroc_list.append(roc_auc_score(test, pred))
	    else:
	        for j in range(len(ns_gw_best_y_test_all_auroc[2][0][0])): # per cv
	            test = ns_gw_best_y_test_all_auroc[i][2][0][j]
	            pred = ns_gw_best_y_pred_all_auroc[i][2][0][j]
	            test_list.append(test)
	            pred_list.append(pred)
	            auroc_list.append(roc_auc_score(test, pred))

	    test_list_gw.append(test_list)
	    pred_list_gw.append(pred_list)
	    auroc_list_gw.append(auroc_list)

	fprs = []
	base_tpr = np.linspace(0, 1, 101)
	plt.figure(figsize=(2.7,2.7))
	color_set = ['violet','darkviolet','rebeccapurple']
	label_set = ['28 weeks', '32 weeks', '37 weeks']
	for gw_idx in range(len(gw_list)):
	    for i in range(5):
	        fpr, tpr, _ = roc_curve(test_list_gw[gw_idx][i], pred_list_gw[gw_idx][i], drop_intermediate=False)
	        plt.plot(fpr, tpr, 'b', alpha=0.1, color=color_set[gw_idx])

	        tpr = interp(base_fpr, fpr, tpr)
	        tpr[0] = 0.0
	        tprs.append(tpr)

	    auc_t = statistics.mean(auroc_list_gw[gw_idx])
	    tprs = np.array(tprs)
	    mean_tprs = tprs.mean(axis=0)
	    std = tprs.std(axis=0)

	    tprs_upper = np.minimum(mean_tprs + std, 1)
	    tprs_lower = mean_tprs - std

	    plt.plot(base_fpr, mean_tprs, 'b', lw=2, color=color_set[gw_idx], marker='', label=label_set[gw_idx]+' (auROC = %0.2f)' % auc_t)

	    tprs = []
	    base_fpr = np.linspace(0, 1, 101)

	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([-0.01, 1.01])
	plt.ylim([-0.01, 1.01])
	# plt.ylabel('True Positive Rate', fontsize=15)
	# plt.xlabel('False Positive Rate', fontsize=15)
	#plt.title("A. Metabolomics Severe sPTB prediction", weight='bold', fontsize=8)
	plt.legend(loc='lower right', fontsize=5)
	plt.savefig('figurePanels/S8C.png', dpi=800)
	plt.savefig('figurePanels/S8C.pdf', dpi=800)
	plt.close()

def figure_S8DE():
	b16 = pd.read_pickle("final_models/validation/val_AA_v2_black_2016.pkl")
	w14 = pd.read_pickle("final_models/validation/val_white_v2_white_2014.pkl")

	b_16_y_pred = np.nan_to_num(-stats.zscore(b16[0][2][0][1].y_pred_test), nan=1)
	b_16_y_test = b16[0][2][0][1].y_test
	w_14_y_pred = np.nan_to_num(-stats.zscore(w14[0][2][0][1].y_pred_test), nan=1)
	w_14_y_test = w14[0][2][0][1].y_test

	fpr_1, tpr_1, thresholds_1 = precision_recall_curve(w_14_y_test, w_14_y_pred)
	auc_1 = average_precision_score(w_14_y_test, w_14_y_pred)

	c=['teal']

	plt.figure(figsize=(2.7,2.7))
	lw = 2
	plt.plot([0.5, 0.5], "r--", lw=lw, linestyle='--', label = 'Class balance (ratio = 0.50)')
	plt.plot(tpr_1, fpr_1, color=c[0], lw=lw, linestyle = '-', label='Ghartey 2015 (auPR = %0.2f)' % auc_1)

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	#plt.xlabel('False Positive Rate', fontsize=15)
	#plt.ylabel('True Positive Rate', fontsize=15)
	plt.legend(loc="lower right", fontsize=5, handlelength=4.1)
	plt.savefig('figurePanels/S8D.pdf', dpi=800)
	plt.savefig('figurePanels/S8D.png', dpi=800)
	plt.close()

	fpr_1, tpr_1, thresholds_1 = precision_recall_curve(b_16_y_test, b_16_y_pred)
	auc_1 = average_precision_score(b_16_y_test, b_16_y_pred)

	c=['olive']

	plt.figure(figsize=(2.7,2.7))
	lw = 2
	plt.plot([0.4, 0.4], 'r--', lw=lw, linestyle='--', label = 'Class balance (ratio = 0.40)')
	plt.plot(tpr_1, fpr_1, color=c[0], lw=lw, linestyle = '-', label='Ghartey 2017 (auPR = %0.2f)' % auc_1)


	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])

	#plt.xlabel('False Positive Rate', fontsize=15)
	#plt.ylabel('True Positive Rate', fontsize=15)
	plt.legend(loc="lower right", fontsize=5, handlelength=4.1)
	plt.savefig('figurePanels/S8E.pdf', dpi=800)
	plt.savefig('figurePanels/S8E.png', dpi=800)
	plt.close()

def figure_S8FG():

	# read shap required files
	res_AA = pd.read_pickle("final_models/shap/elo_False_1_combo_AA_shap.pkl")
	res_non_AA = pd.read_pickle("final_models/shap/elo_False_1_combo_non_AA_shap.pkl")

	full_AA = pd.read_csv("final_models/shap/x_train_Clinical_age_combo_173.csv", index_col=0)
	full_non_AA = pd.read_csv("final_models/shap/x_train_Clinical_age_combo_52.csv", index_col=0)


	af_AA = res_AA[0][2][9][1].shap_values_all
	af_non_AA = res_non_AA [0][2][9][1].shap_values_all

	x_train_AA = pd.DataFrame(res_AA[0][2][9][1].x_train)
	x_train_non_AA = pd.DataFrame(res_non_AA [0][2][9][1].x_train)

	af = af_AA.append(af_non_AA , sort=False).fillna(0)
	x_train = x_train_AA.append(x_train_non_AA, sort=False)

	shap.initjs()

	for row, row_val in x_train.iterrows():
	    for col in row_val.keys():
	        if pd.isnull(x_train.loc[row, col]):
	            if row in af_AA.index:
	                if 'PC' not in col:
	                    x_train.loc[row, col] = full_AA.loc[row, col]
	                else:
	                    x_train.at[row, col] = x_train.loc[:, col].mean()
	            else:
	                try:
	                    x_train.at[row, col] = full_non_AA.loc[row, col]
	                except:
	                    x_train.at[row, col] = 0

	shap.summary_plot(af.values, x_train,
	                  #sort=False,
	                  plot_type="dot",
	                  max_display=10,
	                  show=False,
	                  color_bar=True,
	                  #color_bar_label = True
	                 )

	figure = plt.gca()
	plt.xlabel("")
	y_axis = figure.axes.get_yaxis()
	y_axis.set_visible(False)
	f = plt.gcf()
	f.savefig('figurePanels/S8F.png', dpi=800, figsize = (5,5), bbox_inches = 'tight')
	f.savefig('figurePanels/S8F.pdf', dpi=800, figsize = (5,5), bbox_inches = 'tight')
	plt.close()


	# read shap required files
	res_AA = pd.read_pickle("final_models/shap/micro_AA_shap.pkl")
	res_non_AA = pd.read_pickle("final_models/shap/micro_non_AA_shap.pkl")

	full_AA = pd.read_csv("final_models/shap/x_train_Otu1_173.csv", index_col=0)
	full_non_AA = pd.read_csv("final_models/shap/x_train_Otu1_59.csv", index_col=0)

	af_AA = res_AA[0][2][9][1].shap_values_all
	af_non_AA = res_non_AA [0][2][9][1].shap_values_all

	x_train_AA = pd.DataFrame(res_AA[0][2][9][1].x_train)
	x_train_non_AA = pd.DataFrame(res_non_AA [0][2][9][1].x_train)

	af = af_AA.append(af_non_AA , sort=False).fillna(0)
	x_train = x_train_AA.append(x_train_non_AA, sort=False)

	shap.initjs()

	for row, row_val in x_train.iterrows():
	    for col in row_val.keys():
	        if pd.isnull(x_train.loc[row, col]):
	            if row in af_AA.index:
	                x_train.loc[row, col] = full_AA.loc[row, col]
	            else:
	                try:
	                    x_train.at[row, col] = full_non_AA.loc[row, col]
	                except:
	                    x_train.at[row, col] = 0

	usr_tax = pd.read_csv('otus.sintax', sep = '\t', header = None).set_index(0)[3]\
	    .apply(lambda x: [t.split(':')[1] for t in x.split(',')] if not pd.isnull(x) else np.nan).apply(Series)\
	    .rename(columns = {0:'Kingdom', 1:'Phylum', 2:'Class', 3:'Order', 4:'Family', 5:'Genus', 6:'Species'})

	af = af.rename(columns = lambda r: '%s (%s)' % (r, (['%s: %s' % (hot[0], usr_tax.loc[r, hot].replace('_',' ').replace('\"', '').replace('\\', '')) \
	                                   for hot in ('Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum') \
	                                     if pd.notnull(usr_tax.loc[r, hot])] + ['Unknown'])[0]))

	shap.summary_plot(af.values, x_train,
	                  #sort=False,
	                  plot_type="dot",
	                  max_display=10,
	                  show=False,
	                  color_bar=True,
	                  #color_bar_label = True
	                 )

	figure = plt.gca()
	plt.xlabel("")
	y_axis = figure.axes.get_yaxis()
	y_axis.set_visible(False)
	f = plt.gcf()
	f.savefig('figurePanels/S8G.png', dpi=800, figsize = (5,5), bbox_inches = 'tight')
	f.savefig('figurePanels/S8G.pdf', dpi=800, figsize = (5,5), bbox_inches = 'tight')
	plt.close()
