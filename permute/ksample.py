"""
K-sample permutation tests.
"""


import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.stats import ttest_ind, ttest_1samp

from .utils import get_prng, permute, permute_within_groups


def k_sample(x, group, reps=10**5, stat='one-way anova',
               keep_dist=False, seed=None, plus1=True):
    r"""
    k-sample permutation test for equality of more than 2 means, 
    with p-value estimated by simulated random sampling with
    reps replications.

    Tests the hypothesis that groupings are a random partition of x
    against the alternative that at least one group comes from a 
    population with mean different from the rest

    If ``keep_dist``, return the distribution of values of the test statistic;
    otherwise, return only the number of permutations for which the value of
    the test statistic and p-value.

    Parameters
    ----------
    x : array-like
        Sample values
    group : array-like
        Group labels for each observation
    reps : int
        number of repetitions
    stat : {'one-way anova'}
        The test statistic.

        (a) If stat == 'one-way anova', use the sum of squared 
               distances between the group means and the overall mean
               weighted by group size.
               $\sum_{k=1}^K n_k(\overline{X_k} - \overline{X})^2$
    keep_dist : bool
        flag for whether to store and return the array of values
        of the irr test statistic
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator
    plus1 : bool
        flag for whether to add 1 to the numerator and denominator of the
        p-value based on the empirical permutation distribution. 
        Default is True.

    Returns
    -------
    float
        the estimated p-value
    float
        the test statistic
    list
        The distribution of test statistics.
        These values are only returned if `keep_dist` == True
    """

    prng = get_prng(seed)

    # If stat is callable, use it as the test function. Otherwise, look in the
    # dictionary
    stats = {
        "one-way anova" : one_way_anova
    }
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]

    xbar = np.mean(x)    
    observed_tst = tst_fun(x, group, xbar)

    if keep_dist:
        dist = np.empty(reps)
        for i in range(reps):
            group_perm = permute(group, prng)
            dist[i] = tst_fun(x, group_perm, xbar)
        pvalue = (plus1 + np.sum(dist >= observed_tst))/(plus1 + reps)
        return pvalue, observed_tst, dist
    else:
        hits = 0
        for i in range(reps):
            group_perm = permute(group, prng)
            tst = tst_fun(x, group_perm, xbar)
            if tst >= observed_tst:
                hits += 1
        return (plus1 + hits)/(plus1 + reps), observed_tst


def one_way_anova(x, group, overall_mean):
    r"""
    Test statistic for one-way ANOVA

    Parameters
    ----------
    x : array-like
        Sample values
    group : array-like
        Group labels for each observation
    overall_mean : float
        mean of x

    Returns
    -------
    float
        the one-way ANOVA statistic 
        $\sum_{k=1}^K n_k(\overline{X_k} - \overline{X})^2$
        where $k$ indexes the groups
    """
    x = np.array(x)
    tst = 0
    for k in np.unique(group):
        group_k = x[group == k]
        group_mean = np.mean(group_k)
        nk = len(group_k)
        tst += (group_mean - overall_mean)**2 * nk
    return tst


def bivariate_k_sample(x, group1, group2, reps=10**5, stat='two-way anova',
               keep_dist=False, seed=None, plus1=True):
    r"""
    k-sample permutation test for equality of more than 2 means, 
    with p-value estimated by simulated random sampling with
    reps replications.

    Tests the hypothesis that within grouping 1, grouping 2 is
     a random partition of x against the alternative that at 
    least one group 2 comes from a population with mean different from the rest

    If ``keep_dist``, return the distribution of values of the test statistic;
    otherwise, return only the number of permutations for which the value of
    the test statistic and p-value.

    Parameters
    ----------
    x : array-like
        Sample values
    group1 : array-like
        Fixed group labels for each observation
    group2 : array-like
        Group labels that, under the null, are exchangeable for each 
        level of group1
    reps : int
        number of repetitions
    stat : {'two-way anova'}
        The test statistic.
;
        (a) If stat == 'two-way anova', use a simpler statistic that
               is permutationally equivalent to the F statistic:
               $SSB/(SST - SSB).$
               $SSB$ is the sum of squared deviations from the
               overall mean for the group 2 means and $SST$ is
               the sum of squared deviations of all observations
    keep_dist : bool
        flag for whether to store and return the array of values
        of the irr test statistic
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator
    plus1 : bool
        flag for whether to add 1 to the numerator and denominator of the
        p-value based on the empirical permutation distribution. 
        Default is True.

    Returns
    -------
    float
        the estimated p-value
    float
        the test statistic
    list
        The distribution of test statistics.
        These values are only returned if `keep_dist` == True
    """

    prng = get_prng(seed)

    # If stat is callable, use it as the test function. Otherwise, look in the
    # dictionary
    stats = {
        "two-way anova" : two_way_anova
    }
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]

    xbar = np.mean(x)
    observed_tst = tst_fun(x, group1, group2, xbar)

    if keep_dist:
        dist = np.empty(reps)
        for i in range(reps):
            group2_perm = permute_within_groups(group2, group1, prng)
            dist[i] = tst_fun(x, group1, group2_perm, xbar)
        pvalue = (plus1 + np.sum(dist >= observed_tst))/(plus1 + reps)
        return pvalue, observed_tst, dist
    else:
        hits = 0
        for i in range(reps):
            group2_perm = permute_within_groups(group2, group1, prng)
            tst = tst_fun(x, group1, group2_perm, xbar)
            if tst >= observed_tst:
                hits += 1
        return (plus1 + hits)/(plus1 + reps), observed_tst


def two_way_anova(x, group1, group2, overall_mean):
    """
    Test statistic for two-way ANOVA. The test statistic
    is permutationally equivalent to the F statistic.

    Parameters
    ----------
    x : array-like
        Sample values
    group1 : array-like
        Fixed group labels for each observation
    group2 : array-like
        Group labels that, under the null, are exchangeable for each 
        level of group1
    overall_mean : float
        mean of x

    Returns
    -------
    float
        the statistic $SSB/(SST - SSB),$ where
        $SSB$ is the sum of squared deviations from the
        overall mean for the group 2 means and $SST$ is
        the sum of squared deviations of all observations
    """
    sst = np.sum((x - overall_mean)**2)
    ss2 = 0
    for g in np.unique(group2):
        xx = x[group2 == g]
        ss2 += (np.mean(xx) - overall_mean)**2
    return ss2/(sst - ss2)
