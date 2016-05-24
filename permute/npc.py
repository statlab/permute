from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from scipy.stats import norm, rankdata


try:
    basestring
except NameError:
    basestring = str


# Combining functions

def fisher(pvalues):
    r"""
    Apply Fisher's combining function

    .. math:: -2 \sum_i \log(p_i)

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine

    Returns
    -------
    float
        Fisher's combined test statistic
    """
    return -2 * np.log(np.prod(pvalues))


def liptak(pvalues):
    r"""
    Apply Liptak's combining function

    .. math:: \sum_i \Phi^{-1}(1-p_i)

    where $\Phi^{-1}$ is the inverse CDF of the standard normal distribution.

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine

    Returns
    -------
    float
        Liptak's combined test statistic
    """
    return np.sum(norm.ppf(1 - pvalues))


def tippett(pvalues):
    r"""
    Apply Tippett's combining function

    .. math:: \max_i \{1-p_i\}

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine

    Returns
    -------
    float
        Tippett's combined test statistic
    """
    return np.max(1 - pvalues)


def inverse_n_weight(pvalues, size):
    r"""
    Compute the test statistic

    .. math:: -\sum_{s=1}^S \frac{p_s}{\sqrt{N_s}}

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

def t2p(distr, alternative="greater"):
    r"""
    Use the empirical distribution of a test statistic to compute
    p-values for every value in the distribution.

    Parameters
    ----------
    distr : array_like
        Empirical distribution of statistic
    alternative : {'greater', 'less', 'two-sided'}
        The alternative hypothesis to test (default is 'greater')

    Returns
    -------
    float
        the estimated p-vlaue
    """

    if not alternative in ['greater', 'less', 'two-sided']:
        raise ValueError('Bad alternative')
    B = len(distr)
    if alternative != "less":
        pupper = 1 - (rankdata(distr, method="min") / B) + 1 / B
        pvalue = pupper
    if alternative != "greater":
        plower = rankdata(distr, method="min") / B
        pvalue = plower
    if alternative == "two-sided":
        pvalue = np.min([np.ones(B), 2 * np.min([plower, pupper], 0)], 0)
    return pvalue


def check_combfunc_monotonic(pvalues, combfunc):
    r"""
    Utility function to check that the combining function is monotonically
    decreasing in each argument.

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine
    combine : function
        The combining function to use.

    Returns
    -------
    ``True`` if the combining function passed the check, ``False`` otherwise.
    """

    obs_ts = combfunc(pvalues)
    for i in range(len(pvalues)):
        test_pvalues = pvalues.copy()
        test_pvalues[i] = test_pvalues[i] + 0.1
        if(obs_ts < combfunc(test_pvalues)):
            return False
    return True


def npc(pvalues, distr, combine="fisher", alternatives="greater"):
    r"""
    Combines p-values from individual partial test hypotheses $H_{0i}$ against
    $H_{1i}$, $i=1,\dots,n$ to test the global null hypothesis

    .. math:: \cap_{i=1}^n H_{0i}

    against the alternative

    .. math:: \cup_{i=1}^n H_{1i}

    using an omnibus test statistic.

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine
    distr : array_like
        Array of dimension [B, n] where B is the number of permutations and n is
        the number of partial hypothesis tests. The $i$th column of distr contains
        the simulated null distribution of the $i$th test statistic under $H_{0i}$.
    combine : {'fisher', 'liptak', 'tippett'} or function
        The combining function to use. Default is "fisher".
        Valid combining functions must take in p-values as their argument and be
        monotonically decreasing in each p-value.
    alternatives : array_like
        Optional, an array containing the alternatives for each partial test
        ('greater', 'less', 'two-sided') or a single alternative, if all tests
        have the same alternative hypothesis. Default is "greater".

    Returns
    -------
    float
        A single p-value for the global test
    """
    n = len(pvalues)
    B = distr.shape[0]
    if n < 2:
        raise ValueError("One p-value: nothing to combine!")
    if n != distr.shape[1]:
        raise ValueError("Mismatch in number of p-values and size of distr")
    if isinstance(alternatives, basestring):
        alternatives = np.array([alternatives] * n)
    elif len(alternatives) != n:
        raise ValueError("Mismatch in number of p-values and alternatives")

    combine_library = {
        "fisher": fisher,
        "liptak": liptak,
        "tippett": tippett
    }
    if callable(combine):
        if not check_combfunc_monotonic(pvalues, combine):
            raise ValueError(
                "Bad combining function: must be monotonically decreasing in each p-value")
        combine_func = combine
    else:
        combine_func = combine_library[combine]

    # Convert test statistic distribution to p-values
    combined_stat_distr = [0] * B
    pvalues_from_distr = np.zeros((B, n))
    for j in range(n):
        pvalues_from_distr[:, j] = t2p(distr[:, j], alternatives[j])
    if combine == "liptak":
        toobig = np.where(pvalues_from_distr == 1)
        pvalues_from_distr[toobig] = 0.9999
    combined_stat_distr = np.apply_along_axis(
        combine_func, 1, pvalues_from_distr)

    observed_combined_stat = combine_func(pvalues)
    return np.sum(combined_stat_distr >= observed_combined_stat) / B
