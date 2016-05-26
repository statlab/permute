"""
Stratified permutation tests.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import math

from .utils import get_prng, permute_within_groups


def corrcoef(x, y, group):
    r"""
    Calculates sum of Spearman correlations between x and y,
    computed separately in each group.

    Parameters
    ----------
    x : array-like
        Variable 1
    y : array-like
        Variable 2, of the same length as x
    group : array-like
        Group memberships, of the same length as x

    Returns
    -------
    float
        The sum of Spearman correlations
    """
    tst = 0.0
    for g in np.unique(group):
        gg = group == g
        tst += np.corrcoef(x[gg], y[gg])[0, 1]

    return tst


def sim_corr(x, y, group, reps=10**4, alternative='greater', seed=None):
    r"""
    Simulate permutation p-value of stratified Spearman correlation test.

    Parameters
    ----------
    x : array-like
        Variable 1
    y : array-like
        Variable 2, of the same length as x
    group : array-like
        Group memberships, of the same length as x
    alternative : {'greater', 'less', 'two-sided'}
        The alternative hypothesis to test
    reps : int
        Number of repetitions
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator.

    Returns
    -------
    float
      the estimated p-value
    float
      the observed test statistic
    list
      the null distribution
    """
    prng = get_prng(seed)
    tst = corrcoef(x, y, group)
    dist = [corrcoef(permute_within_groups(x, group, prng), y, group)
            for i in range(reps)]
    right_pv = np.sum(dist >= tst) / reps
    
    thePvalue = {
        'greater': lambda p: p,
        'less': lambda p: 1 - p,
        'two-sided': lambda p: 2 * np.min([p, 1 - p])
    }
    return thePvalue[alternative](right_pv), tst, dist


def stratified_permutationtest_mean(group, condition, response,
                                    groups=None, conditions=None):
    r"""
    Calculates variability in sample means between treatment conditions,
    within groups.

    If there are two treatment conditions, the test statistic is the
    difference in means, aggregated across groups.
    If there are more than two treatment conditions, the test statistic
    is the standard deviation of the means, aggregated across groups.

    Parameters
    ----------
    group : array-like
        Group memberships
    condition : array-like
        Treatment conditions, of the same length as group
    response : array-like
        Responses, of the same length as group
    groups : array-like
        Group labels. By default, it is the unique values of group
    conditions : array-like
        Condition labels. By default, it is the unique values of condition


    Returns
    -------
    tst : float
      The observed test statistic
    """
    if groups is None:
        groups = np.unique(group)
    if conditions is None:
        conditions = np.unique(condition)

    tst = 0.0
    if len(groups) < 2:
        raise ValueError('Number of groups must be at least 2.')
    elif len(groups) == 2:
        stat = lambda u: math.fabs(u[0] - u[1])
    elif len(groups) > 2:
        stat = np.std
    for g in groups:
        gg = group == g
        x = [gg & (condition == c) for c in conditions]
        tst += stat([response[x[j]].mean() for j in range(len(x))])
    return tst


def stratified_permutationtest(group, condition, response, alternative='greater',
                               reps=10**5, testStatistic='mean',
                               seed=None):
    r"""
    Stratified permutation test based on differences in means.

    The test statistic is

    .. math:: \sum_{g \in \text{groups}} [
                 f(mean(\text{response for cases in group $g$
                               assigned to each condition}))].

    The function f is the difference if there are two conditions, and
    the standard deviation if there are more than two conditions.

    There should be at least one group and at least two conditions.
    Under the null hypothesis, all assignments to the two conditions that
    preserve the number of cases assigned to the conditions are equally
    likely.

    Groups in which all cases are assigned to the same condition are
    skipped; they do not contribute to the p-value since all randomizations
    give the same contribution to the difference in means.

    Parameters
    ----------
    group : array-like
        Group memberships
    condition : array-like
        Treatment conditions, of the same length as group
    response : array-like
        Responses, of the same length as group
    alternative : {'greater', 'less', 'two-sided'}
        The alternative hypothesis to test
    reps : int
        Number of repetitions
    testStatistic : function
        Function to compute test statistic. By default, stratified_permutationtest_mean
        The test statistic. Either a string or function.

        (a) If stat == 'mean', the test statistic is stratified_permutationtest_mean (default).
        (b) If stat is a function (a callable object), the test statistic is
            that function.  The function should take a permutation of the
            data and compute the test function from it. For instance, if the
            test statistic is the maximum absolute value, $\max_i |z_i|$,
            the test statistic could be written:

            f = lambda u: np.max(abs(u))
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator

    Returns
    -------
    float
      the estimated p-value
    float
      the observed test statistic
    list
      the null distribution
    """
    prng = get_prng(seed)
    groups = np.unique(group)
    conditions = np.unique(condition)
    
    stats = {
        'mean': lambda u: stratified_permutationtest_mean(group, u,
                                    response, groups, conditions)
    }
    if callable(testStatistic):
        tst_fun = testStatistic
    else:
        tst_fun = stats[testStatistic]
    thePvalue = {
        'greater': lambda p: p,
        'less': lambda p: 1 - p,
        'two-sided': lambda p: 2 * np.min([p, 1 - p])
    }
    
    if len(conditions) < 2:
        return 1.0, np.nan, None
    else:
        tst = tst_fun(condition)
        dist = np.zeros(reps)
        for i in range(int(reps)):
            dist[i] = tst_fun(permute_within_groups(condition, group, prng))

        right_pv = np.sum(dist >= tst) / reps
        return thePvalue[alternative](right_pv), tst, dist
