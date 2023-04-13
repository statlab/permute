import numpy as np

def alpha_martingale(x: np.ndarray, mu_null: float, upper_bound: float, eta_j_func: callable) -> float:
    """Implementation of the ALPHA martingale described in
    https://arxiv.org/pdf/2201.02707.pdf

    Args:
        x (np.ndarray): Input data to be read in a streaming fashion.
        mu_null (float): Mean under the null hypothesis.
        upper_bound (float): Deterministic upper bound for all data in the underlying distribution.
        eta_j_func (callable): Function that obtains eta_j, as described in the paper above.

    Returns:
        float: Anytime p-value.
    """
    anytime_pval = 1.0
    T_j = 1.0
    for j in range(len(x)):
        x_j = x[:j+1]
        eta_j = eta_j_func(x_j[:-1])
        print('eta_j', eta_j)
        assert eta_j >= mu_null and eta_j <= upper_bound, 'eta_j not in correct range'
        T_j *= (x_j[-1]/mu_null) * ((eta_j - mu_null)/(upper_bound - mu_null)) + ((upper_bound - eta_j)/(upper_bound - mu_null))
        anytime_pval = min(anytime_pval, 1/max(1.0,T_j))
    return anytime_pval

def sample_mean_wor_func(N: int, mu_null: float, upper_bound: float, d: float=10) -> callable:
    """Compute sample mean assuming no replacement, using truncated shrinkage
    from https://arxiv.org/pdf/2201.02707.pdf.

    Args:
        N (int): Total number of elements in population.
        mu_null (float): Mean under the null hypothesis.
        upper_bound (float): Deterministic upper bound for all data in the underlying distribution.
        d (float, optional): Hyperparameter from the ALPHA paper. Defaults to 10.
    
    Returns:
        callable: Function to compute eta_j given data up to timestep j-1.
    """
    def sample_mean_wor_eta(x_jm1):
        if len(x_jm1) == 0:
            eta_j = mu_null
        else:
            eta_j = (N*mu_null - np.sum(x_jm1))/(N + d - len(x_jm1))
        return np.clip(eta_j, mu_null + 1/np.sqrt(d + len(x_jm1)), upper_bound)
    return sample_mean_wor_eta

def sample_mean_func(mu_null: float, upper_bound: float, d: float=10) -> callable:
    """Compute sample mean assuming replacement, using truncated shrinkage
    from https://arxiv.org/pdf/2201.02707.pdf.

    Args:
        mu_null (float): Mean under the null hypothesis.
        upper_bound (float): Deterministic upper bound for all data in the underlying distribution.
        d (float, optional): Hyperparameter from the ALPHA paper. Defaults to 10.
    
    Returns:
        callable: Function to compute eta_j given data up to timestep j-1.
    """
    def sample_mean_eta(x_jm1):
        jm1 = len(x_jm1.flatten())
        if len(x_jm1) == 0:
            eta_j = mu_null
        else:
            eta_j = (d*mu_null + np.sum(x_jm1))/(d + jm1)
        return np.clip(eta_j, mu_null + 1/np.sqrt(d + jm1), upper_bound)
    return sample_mean_eta

def sample_mean_naive_func(mu_null: float, upper_bound: float, eps: float=0.1, data_threshold: int=0) -> callable:
    """Compute sample mean assuming replacement, using truncation without shrinkage.

    Args:
        mu_null (float): Mean under the null hypothesis.
        upper_bound (float): Deterministic upper bound for all data in the underlying distribution.
        eps (float, optional): Fixed epsilon for truncation. Defaults to 0.1.
        data_threshold (int, optional): Number of timesteps until you use data to estimate eta_j. Defaults to 0.

    Returns:
        callable: Function to compute eta_j given data up to timestep j-1.
    """
    def sample_mean_eta(x_jm1):
        if len(x_jm1) <= data_threshold:
            eta_j = mu_null
        else:
            eta_j = (mu_null + np.sum(x_jm1))/(1 + len(x_jm1))
        return np.clip(eta_j, mu_null + eps*(upper_bound - mu_null), upper_bound)
    return sample_mean_eta

def sample_mean_wor_naive_func(N: int, mu_null: float, upper_bound: float, eps: float=0.1, data_threshold: int=0) -> callable:
    """Compute sample mean assuming no replacement, using truncation without shrinkage.

    Args:
        N (int): Total number of elements in population.
        mu_null (float): Mean under the null hypothesis.
        upper_bound (float): Deterministic upper bound for all data in the underlying distribution.
        eps (float, optional): Fixed epsilon for truncation. Defaults to 0.1.
        data_threshold (int, optional): Number of timesteps until you use data to estimate eta_j. Defaults to 0.

    Returns:
        callable: Function to compute eta_j given data up to timestep j-1.
    """
    def sample_mean_wor_eta(x_jm1):
        if len(x_jm1) <= data_threshold:
            eta_j = mu_null
        else:
            eta_j = (N*mu_null - np.sum(x_jm1))/(N + 1 - len(x_jm1))
        return np.clip(eta_j, mu_null + eps*(upper_bound - mu_null), upper_bound)
    return sample_mean_wor_eta

def lambda_predictable_plugin_func(mu_null: float, upper_bound: float, significance_level: float) -> callable:
    """Compute the predictable plugin lambda described in eq. (26) of
    https://arxiv.org/pdf/2010.09686.pdf and derive eta_j from algebraic manipulation.

    Args:
        mu_null (float): Mean under the null hypothesis.
        upper_bound (float): Deterministic upper bound for all data in the underlying distribution.
        significance_level (float): Significance level of hypothesis test.

    Returns:
        callable: Function to compute eta_j given data up to timestep j-1.
    """
    def lambda_predictable_plugin(x_jm1):
        t = len(x_jm1)+1
        if len(x_jm1) == 0:
            sigma_tm1 = 0.25/t
        else:
            mu_hats = (np.cumsum(x_jm1) + 0.5)/(np.array(range(2, t+1)))
            sigma_tm1 = (np.sum(np.square(x_jm1 - mu_hats)) + 0.25)/t
        lambda_t = np.sqrt(2*np.log(2/significance_level)/(sigma_tm1**2 * t * np.log(t+1)))
        lambda_j = np.clip(np.abs(lambda_t), mu_null, upper_bound)
        eta_j = (lambda_j*(upper_bound - mu_null) + 1)*mu_null
        return eta_j
    return lambda_predictable_plugin

def bernoulli_bayesian_beta_prior_func(mu_null: float, a_init: float, b_init: float, upper_bound: float, eps: float=0.1, data_threshold: int=0) -> callable:
    """Estimate parameter of a Bernoulli distribution using a
    Beta distribution prior (take the maximum likelihood estimate).
    The parameters of the prior are updated using Bayesian updating.

    Args:
        mu_null (float): Mean (Bernoulli parameter) under the null hypothesis.
        a_init (float): Initial guess for 'a' parameter in Beta distribution.
        b_init (float): Initial guess for 'b' parameter in Beta distribution.
        upper_bound (float): Deterministic upper bound for all data in the underlying distribution.
        eps (float, optional): Fixed epsilon for truncation. Defaults to 0.1.
        data_threshold (int, optional): Number of timesteps until you use data to estimate eta_j. Defaults to 0.

    Returns:
        callable: Function to compute eta_j given data up to timestep j-1.
    """
    a = a_init
    b = b_init
    assert a_init > 0 and b_init > 0, 'shape parameters must be positive'
    def bernoulli_bayesian_beta_prior(x_jm1):
        if len(x_jm1) <= data_threshold:
            eta_j = mu_null
        else:
            nonlocal a
            nonlocal b
            x_last = x_jm1[-1]
            a += x_last
            b += 1 - x_last
            if a == 1 and b == 1:
                ml_mu = 0.5
            else:
                ml_mu = (a - 1)/(a + b - 2)
            eta_j = ml_mu
        return np.clip(eta_j, mu_null + eps*(upper_bound - mu_null), upper_bound)
    return bernoulli_bayesian_beta_prior