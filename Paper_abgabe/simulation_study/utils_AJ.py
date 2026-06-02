import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from class_DecisionTreeBaggingClassifier import DecisionTreeBaggingClassifier
import os, json
import matplotlib.pyplot as plt
import warnings
from utils import create_weibull_data, stratified_split, create_data_with_ipc_weights, ipc_weighted_mse, calculate_bootstrap_variance



#########################################################################################################
' contributions to variance estimator '
    
def calculate_contributions_variance(
    clf: DecisionTreeBaggingClassifier,
    X_pred_point: pd.DataFrame,
    df_train: pd.DataFrame
): 

    T_N_b, pred = clf.predict_proba(X_pred_point.values)
    N_bi = clf.nbi
    weights = df_train["weights_ipcw"]

    B, n = N_bi.shape

    n_plus = np.sum(weights > 0)

    TNb = T_N_b
    Nbi = N_bi
    
    cov_i = ((N_bi - n * weights.values.reshape(1,-1)).T @ (T_N_b - pred)) / B
    cov_i_hoch2 = cov_i**2

    cov_i_wager = ((N_bi - 1).T @ (T_N_b - pred)) / B
    cov_i_wager_hoch2 = cov_i_wager**2

    array = cov_i_hoch2 / ((weights.values.reshape(-1,1))**2)
    
    hatV = np.var(T_N_b, axis=0, ddof=1)
    
    sumw = np.sum( ( 1 / (weights[weights > 0] ) ) -1) 

    return (
        B,
        n,
        n_plus,
        TNb, 
        Nbi,
        weights,
        cov_i,
        cov_i_wager,
        array,
        hatV,
        sumw,
        df_train
    )

   



#########################################################################################################
' Single Simulation function '
def simulationsingle(
    seed: int,
    data_generation_parameter: dict,
    params_rf: dict):

    data = create_weibull_data(params=data_generation_parameter, random_state=seed)
    df_train, df_test = stratified_split(data=data, random_state=seed, test_size=0.3)
    df_train = create_data_with_ipc_weights(data=df_train, params=data_generation_parameter)
    df_test = create_data_with_ipc_weights(data=df_test, params=data_generation_parameter)

    ### Random Forest Modell ###
    # Fit
    params_rf["random_state"] = seed
    clf = DecisionTreeBaggingClassifier(params_rf)
    clf.fit(
            X=df_train.drop(
                ["time", "event", "weights_ipcw", "survived"], axis=1
            ).values,
            y=df_train["survived"].values,
            sample_weights=df_train["weights_ipcw"].values,
    )

    # Evaluation auf Testdaten
    _ , pred  =clf.predict_proba(df_test.drop(
           ["weights_ipcw", "survived", "time", "event"], axis=1
    ).values)
    rf_test_mse = ipc_weighted_mse(
            y_true=df_test["survived"].values,
            y_pred=pred,
            sample_weight=df_test["weights_ipcw"].values,
    )

    # Prediction für X_erwartung
    _ ,rf_pred = clf.predict_proba(data_generation_parameter['X_pred_point'].values)


    B, n, n_plus, weights, cov_i, cov_i_wager, array, hatV, sumw, df_train = calculate_contributions_variance(
    clf=clf,
    X_pred_point=data_generation_parameter['X_pred_point'],
    df_train=df_train
    )

    return (
        B,
        n,
        n_plus,
        weights,
        cov_i,
        cov_i_wager,
        array,
        hatV,
        sumw,
        df_train
    )
