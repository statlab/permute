from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.stats import norm


# Combining functions

def fisher(pvalues):
    '''
    Apply Fisher's combining function

    .. math:: \-2\sum_i \log(p_i)

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine

    Returns
    -------
    float
        Fisher's combined test statistic
    '''
    return -2 * np.log(np.prod(pvalues))


def liptak(pvalues):
    '''
    Apply Liptak's combining function

    .. math:: \\sum_i \Phi^{-1}(1-p_i)

    where $\Phi^{-1}$ is the inverse CDF of the standard normal distribution.

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine

    Returns
    -------
    float
        Liptak's combined test statistic
    '''
    return np.sum(norm.ppf(1 - pvalues))


def tippett(pvalues):
    '''
    Apply Tippett's combining function

    .. math:: \\max_i \{1-p_i\}

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine

    Returns
    -------
    float
        Tippett's combined test statistic
    '''
    return np.max(1 - pvalues)


def inverse_n_weight(pvalues, size):
    """
    Compute the test statistic

    .. math:: -\\sum_{s=1}^S\\frac{p_s}{\sqrt{N_s}}

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine
    size : array_like
        The $i$th entry is the sample size used for the $i$th test

    Returns
    -------
    float
        combined test statistic
    """
    weights = size ** (-1 / 2)
    return np.sum(-1 * pvalues * weights)


# Nonparametric combination of tests

def t2p(stat, distr, alternative="greater"):
    '''
    Use the empirical distribution of a test statistic to compute
    its p-value.

    Parameters
    ----------
    stat : float
        Test statistic
    distr : array_like
        Empirical distribution of statistic
    alternative : {'greater', 'less', 'two-sided'}
        The alternative hypothesis to test (default is 'greater')

    Returns
    -------
    float
        the estimated p-vlaue
    '''

    B = len(distr)
    if alternative != "less":
        pupper = np.sum(distr >= stat) / B
        pvalue = pupper
    if alternative != "greater":
        plower = np.sum(distr <= stat) / B
        pvalue = plower
    if alternative == "two-sided":
        pvalue = 2 * min([pupper, plower])
    return pvalue


def npc(pvalues, distr, combine="fisher", alternatives="greater"):
    '''
    Combines p-values from individual partial test hypotheses $H_{0i}$ against
    $H_{1i}$, $i=1,\dots,n$ to test the global null hypothesis

    .. math:: \\cap_{i=1}^n H_{0i}

    against the alternative

    .. math:: \\cup_{i=1}^n H_{1i}

    using an omnibus test statistic.

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine
    distr : array_like
        Array of dimension [B, n] where B is the number of permutations and n is
        the number of partial hypothesis tests. The $i$th column of distr contains
        the simulated null distribution of the $i$th test statistic under $H_{0i}$.
    combine : {'fisher', 'liptak', 'tippett'}
        The combining function to use. Default is "fisher"
        TODO: allow user to pass in their own combining function; check that it
        satifies the correct monotonicity. Describe the monotonicity!
    alternatives : array_like
        Optional, an array containing the alternatives for each partial test
        ('greater', 'less', 'two-sided') or a single alternative, if all tests
        have the same alternative hypothesis. Default is "greater".

    Returns
    -------
    float
        A single p-value for the global test
    '''
    n = len(pvalues)
    B = distr.shape[0]
    if n < 2:
        raise ValueError("One p-value: nothing to combine!")
    if n != distr.shape[1]:
        raise ValueError("Mismatch in number of p-values and size of distr")
    if isinstance(alternatives, str):
        alternatives = np.array([alternatives] * n)
    elif len(alternatives) != n:
        raise ValueError("Mismatch in number of p-values and alternatives")
    # check values of alternatives

    combine_library = {
        "fisher": fisher,
        "liptak": liptak,
        "tippett": tippett
    }
    if callable(combine):
        combine_func = combine
    else:
        combine_func = combine_library[combine]

    # Convert test statistic distribution to p-values
    combined_stat_distr = [0] * B
    for b in range(B):
        pvalues_from_distr = [0] * n
        for j in range(n):
            pvalues_from_distr[j] = t2p(
                distr[b, j], distr[:, j], alternatives[j])
        pvalues_from_distr = np.array(pvalues_from_distr)
        if combine == "liptak":
            toosmall = np.where(pvalues_from_distr == 0)
            pvalues_from_distr[toosmall] = 0.0001
            toobig = np.where(pvalues_from_distr == 1)
            pvalues_from_distr[toobig] = 0.9999
        combined_stat_distr[b] = combine_func(pvalues_from_distr)

    observed_combined_stat = combine_func(pvalues)
    return np.sum(combined_stat_distr >= observed_combined_stat) / B
