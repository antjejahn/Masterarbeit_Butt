import numpy as np


def simulate_mean(i, n, B, seed):
    np.random.seed(seed + i)
    x = np.random.normal(0, 1, n)
    
    # Jackknife variance
    var_jackknife = np.var(x, ddof=1) / n
    
    # Bootstrap variance
    mean = np.array([np.mean(np.random.choice(x, n, replace=True)) for _ in range(B)])
    var_boot = np.var(mean, ddof=1)
    return var_jackknife, var_boot



################################################################################################

def jackknife_corr(x, y, func):
    n = len(x)
    idx = np.arange(n)
    jack_i = [func(x[idx != i], y[idx != i])[0, 1] for i in range(n)]
    jack_mean = np.mean(jack_i)
    return ((n - 1) / n) * np.sum((jack_i - jack_mean) ** 2)

def bootstrap_corr(x, y, func, B):
    n = len(x)
    idxs = np.random.choice(np.arange(n), (B, n), replace=True)
    bootstrap = [func(x[idx], y[idx])[0, 1] for idx in idxs]
    return np.var(bootstrap, ddof=1)

def simulate_pearson(mean, cov, n, B, sim_i, seed):
    np.random.seed(seed + sim_i)
    x, y = np.random.multivariate_normal(mean, cov, n).T
    var_jackknife = jackknife_corr(x, y, np.corrcoef)
    var_boot = bootstrap_corr(x, y, np.corrcoef, B)
    return var_jackknife, var_boot

################################################################################################


def jackknife_median(x, func):
    n = len(x)
    idx = np.arange(n)
    jack_i = [func(x[idx != i]) for i in range(n)]
    jack_mean = np.mean(jack_i)
    return ((n - 1) / n) * np.sum((jack_i - jack_mean) ** 2)

def bootstrap_median(x, func, B):
    n = len(x)
    idxs = np.random.choice(np.arange(n), (B, n), replace=True)
    bootstrap = [func(x[idx]) for idx in idxs]
    return np.var(bootstrap, ddof=1)


def simulate_median( n, B, sim_i, seed):
    np.random.seed(seed + sim_i)
    x = np.random.normal(0, 1, n)
    var_jackknife = jackknife_median(x=x, func=np.median)
    var_boot = bootstrap_median(x=x, func=np.median, B=B)
    var_emp = np.median(x)
    return var_jackknife, var_boot,var_emp

