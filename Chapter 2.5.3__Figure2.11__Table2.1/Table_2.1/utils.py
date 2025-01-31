import numpy as np
import pandas as pd

def create_bootstrap_indices_and_Nbi(
    n: int, B: int, seed: int = None, weights: np.ndarray = None
):
    if weights is None:
        rng = np.random.default_rng(seed)
        boot_indices = rng.choice(np.arange(n), size=(B, n), replace=True)
        boot_counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n), axis=1, arr=boot_indices
        )
        return boot_indices, boot_counts

    else:
        rng = np.random.default_rng(seed)
        boot_indices = rng.choice(np.arange(n), size=(B, n), p=weights, replace=True)
        boot_counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n), axis=1, arr=boot_indices
        )
        return boot_indices, boot_counts


def bagging_mean_estimators(x, B, seed, weights):
    n = x.shape[0]
    T_N_b = np.zeros(B)
    indices_list, N_bi = create_bootstrap_indices_and_Nbi(
        n=n, B=B, seed=seed, weights=weights
    )

    for b in range(B):
        indices = indices_list[b]
        T_N_b[b] = np.mean(x[indices])

    return T_N_b, N_bi, indices_list

def simulate_bagging_and_ijk_var_calculation(x1, B, seed, sim_i, weights, m):
    T_N_b, N_bi, boot_indices = bagging_mean_estimators(x=x1, B=B, seed=seed + sim_i, weights=weights)


    ijk_jahn_var, ijk_jahn_var_bias_correction = calculate_ijk_jahn_variance(N_bi=N_bi, T_N_b=T_N_b, weights=weights)
    ijk_wager_var, ijk_wager_var_bias_correction = calculate_ijk_wager_variance(N_bi=N_bi, T_N_b=T_N_b)
    jk_wager_var, jk_wager_var_bias_correction = calculate_jk_varaince(T_N_b=T_N_b, N_bi=N_bi, boot_indices=boot_indices)

    return ijk_jahn_var, ijk_jahn_var_bias_correction, ijk_wager_var, ijk_wager_var_bias_correction, jk_wager_var, jk_wager_var_bias_correction

### var estimators

def calculate_ijk_jahn_variance( T_N_b: np.ndarray, N_bi: np.ndarray, weights: np.ndarray
) -> float:

    pred = T_N_b.mean()
    B, n = N_bi.shape
    n_plus = np.sum(weights > 0)

    cov_i = ((N_bi - n * weights.reshape(1,-1)).T @ (T_N_b - pred)) / B
    cov_i_hoch2 = cov_i**2
    array = cov_i_hoch2/((weights)**2)

    biased_var_estimate = np.sum(array[~np.isnan(array) & ~np.isinf(array)], axis=0) * (1/(np.sum(weights > 0))**2)

    #bias_correction1
    bias_correction =  (1/n_plus**2)  * np.var(T_N_b, axis=0, ddof=1)* n / B * np.sum( ( 1 / (weights[weights > 0] ) ) -1) 

    return biased_var_estimate , bias_correction

def calculate_ijk_wager_variance(T_N_b: np.ndarray, N_bi: np.ndarray) -> float:

    pred = T_N_b.mean()
    B, n = N_bi.shape

    cov_i = ((N_bi - 1).T @ (T_N_b - pred)) / B
    cov_i_hoch2 = cov_i**2

    biased_var_estimate = np.sum(cov_i_hoch2) 

    #bias_correction1
    bias_correction = n / B * np.var(T_N_b, axis=0, ddof=1)

    return biased_var_estimate , bias_correction

def inf_JK_bagged_variance_weighted(N_bi, T_N_b, weights,m) :
    B, n = N_bi.shape
    T_N_b_mean = np.mean(T_N_b, axis=0)
    

    cov_i = ((N_bi - n * weights).T @ (T_N_b - T_N_b_mean)) / B
    cov_i_hoch2 = cov_i**2
    array = cov_i_hoch2/weights

    biased_var_estimate = np.sum(array[~np.isnan(array) & ~np.isinf(array)], axis=0) * np.sum(weights**2)

    bias_correction = n / B * np.sum(1-weights[weights > 0]) * np.var(T_N_b, axis=0, ddof=1)* np.sum(weights**2)

    return biased_var_estimate, bias_correction


def calculate_jk_varaince(T_N_b, N_bi, boot_indices):
    B, n_samples = N_bi.shape
    theta = T_N_b.mean()


    # Prepare a boolean mask for each sample's presence in each estimator's bootstrap
    presence_mask = np.zeros((n_samples, B), dtype=bool)
    for i, samples in enumerate(boot_indices):
        samples = np.array(samples, dtype=int)
        presence_mask[samples, i] = True

    theta_is = []
    for ii in range(n_samples):
        indices_without_ii = np.where(~presence_mask[ii])[0]
        if 0 < len(indices_without_ii) < B:
            theta_is.append(T_N_b[indices_without_ii].mean())

    theta_is = np.array(theta_is)
    var_jka_biased = np.sum((theta_is - theta) ** 2) * (n_samples - 1) / n_samples

    var_jka_correction = (
    (np.exp(1) - 1)
    * (n_samples / B)
    * np.var(T_N_b, ddof=1)
    )

    return var_jka_biased, var_jka_correction