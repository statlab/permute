import numpy as np
from ..alpha import (
    alpha_martingale,
    sample_mean_func,
    sample_mean_wor_func,
    sample_mean_naive_func,
    sample_mean_wor_naive_func,
    lambda_predictable_plugin_func,
    bernoulli_bayesian_beta_prior_func
)

def test_sample_mean_alpha():
    """
    Test truncated shrinkage sample mean estimator (assuming replacement).
    """
    mu_nulls = [0.1, 0.2, 0.8, 0.9]
    reject = [True, True, False, False]
    N = 100
    upper_bound = 1.0
    x = np.random.binomial(1, 0.5, size=(N,))
    for i in range(len(mu_nulls)):
        mu_null = mu_nulls[i]
        eta_j_func = sample_mean_func(mu_null, upper_bound)
        pval = alpha_martingale(x, mu_null, upper_bound, eta_j_func)
        assert (pval < 0.05) == reject[i]

def test_sample_mean_wor_alpha():
    """
    Test truncated shrinkage sample mean estimator (assuming no replacement).
    """
    mu_nulls = [0.1, 0.2, 0.8, 0.9]
    reject = [True, True, False, False]
    N = 100
    upper_bound = 1.0
    x = np.random.choice(range(200), size=(N,), replace=False)/200
    for i in range(len(mu_nulls)):
        mu_null = mu_nulls[i]
        eta_j_func = sample_mean_wor_func(N, mu_null, upper_bound)
        pval = alpha_martingale(x, mu_null, upper_bound, eta_j_func)
        assert (pval < 0.05) == reject[i]

def test_sample_mean_naive_alpha():
    """
    Test truncated sample mean estimator without shrinkage (assuming replacement).
    """
    mu_nulls = [0.1, 0.2, 0.8, 0.9]
    reject = [True, True, False, False]
    N = 100
    upper_bound = 1.0
    x = np.random.binomial(1, 0.5, size=(N,))
    for i in range(len(mu_nulls)):
        mu_null = mu_nulls[i]
        eta_j_func = sample_mean_naive_func(mu_null, upper_bound, data_threshold=10)
        pval = alpha_martingale(x, mu_null, upper_bound, eta_j_func)
        assert (pval < 0.05) == reject[i]

def test_sample_mean_wor_naive_alpha():
    """
    Test truncated sample mean estimator without shrinkage (assuming no replacement).
    """
    mu_nulls = [0.1, 0.2, 0.8, 0.9]
    reject = [True, True, False, False]
    N = 100
    upper_bound = 1.0
    x = np.random.choice(range(200), size=(N,), replace=False)/200
    for i in range(len(mu_nulls)):
        mu_null = mu_nulls[i]
        eta_j_func = sample_mean_wor_naive_func(N, mu_null, upper_bound, data_threshold=10)
        pval = alpha_martingale(x, mu_null, upper_bound, eta_j_func)
        assert (pval < 0.05) == reject[i]

def test_wsr_alpha():
    """
    Test lambda predictable plugin estimator described in Waudby-Smith and Ramdas paper.
    """
    mu_nulls = [0.1, 0.2, 0.8, 0.9]
    reject = [True, True, False, False]
    N = 100
    upper_bound = 1.0
    x = np.random.binomial(1, 0.5, size=(N,))
    for i in range(len(mu_nulls)):
        mu_null = mu_nulls[i]
        eta_j_func = lambda_predictable_plugin_func(mu_null, upper_bound, 0.05)
        pval = alpha_martingale(x, mu_null, upper_bound, eta_j_func)
        assert (pval < 0.05) == reject[i]

def test_bayesian_alpha():
    """
    Test Bayesian updating with Beta prior estimator.
    """
    mu_nulls = [0.1, 0.2, 0.8, 0.9]
    reject = [True, True, False, False]
    N = 100
    upper_bound = 1.0
    x = np.random.binomial(1, 0.5, size=(N,))
    for i in range(len(mu_nulls)):
        mu_null = mu_nulls[i]
        eta_j_func = bernoulli_bayesian_beta_prior_func(mu_null, 1, 1, upper_bound, data_threshold=10)
        pval = alpha_martingale(x, mu_null, upper_bound, eta_j_func)
        assert (pval < 0.05) == reject[i]