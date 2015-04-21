# -*- coding: utf-8 -*-

"""
Stratified permutation tests.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.random import RandomState

from .core import permute_within_groups


def corrcoef(x, y, group):
    """
    Calculates sum of Spearman correlations between x and y,
    computed separately in each group.

    Parameters
    ----------
    x : array-like
    y : array-like
    group : array-like

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


def sim_corr(x, y, group, reps=10**4, prng=None):
    """
    Simulate permutation p-value of stratified Spearman correlation test.

    Parameters
    ----------
    x : array-like
    y : array-like
    group : array-like
    reps : int
    prng : RandomState instance or None, optional (default=None)
        If RandomState instance, prng is the pseudorandom number generator;
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`.

    Returns
    -------
    tuple
        Returns test statistic, simulations, left-sided p-value,
        right-sided p-value, two-sided p-value
    """
    t = corrcoef(x, y, group)
    sims = [corrcoef(permute_within_groups(x, group, prng), y, group)
            for i in range(reps)]
    left_pv = np.sum(sims <= t)/reps
    right_pv = np.sum(sims >= t)/reps
    two_sided_pv = np.sum(np.abs(sims) >= np.abs(t))/reps
    return t, left_pv, right_pv, two_sided_pv, sims


def stratified_permutationtest_mean(group, condition, response,
                                    groups, conditions):
    """
    Calculates variability in sample means between treatment conditions,
    within groups.

    If there are two treatment conditions, the test statistic is the
    difference in means, aggregated across groups.
    If there are more than two treatment conditions, the test statistic
    is the standard deviation of the means, aggregated across groups.

    Parameters
    ----------
    group : int
      The

    Returns
    -------
    tst : float
      The
    """
    tst = 0.0
    if len(groups) < 2:
        raise ValueError('Number of groups must be at least 2.')
    elif len(groups) == 2:
        stat = lambda u: u[0] - u[1]
    elif len(groups) > 2:
        stat = np.std
    for g in groups:
        gg = group == g
        x = [gg & (condition == c) for c in conditions]
        tst += stat([response[x[j]].mean() for j in range(len(x))])
    return tst


def stratified_permutationtest(group, condition, response, iterations=1.0e4,
                               testStatistic=stratified_permutationtest_mean,
                               seed=None):
    """
    Stratified permutation test using the sum of the differences in means
    between two or more conditions in each group (stratum) as the test
    statistic.

    The test statistic is

    .. math:: \sum_{g \in \\text{groups}} [
                 f(mean(\\text{response for cases in group $g$
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
    group : int
      The

    Returns
    -------
    permuted : int
      The
    """
    prng = RandomState(seed)
    groups = np.unique(group)
    conditions = np.unique(condition)
    if len(conditions) < 2:
        return 1.0, 1.0, 1.0, np.nan, None
    else:
        tst = testStatistic(group, condition, response, groups, conditions)
        dist = np.zeros(iterations)
        for i in range(int(iterations)):
            dist[i] = testStatistic(group,
                                    permute_within_groups(
                                        condition, group, prng),
                                    response, groups, conditions)

        conds = [dist <= tst, dist >= tst, abs(dist) >= abs(tst)]
        pLeft, pRight, pBoth = [np.count_nonzero(c)/iterations for c in conds]
        return pLeft, pRight, pBoth, tst, dist
