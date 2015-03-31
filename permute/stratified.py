# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import math

import numpy as np
from numpy.random import RandomState

from scipy.stats import (binom,
                         ttest_ind)
from scipy.optimize import brentq


def permute_within_groups(group, condition, groups, seed=None):
    """
    Permutation of a multiset.

    Parameters
    ----------
    group : int
      The
    condition : int
      The
    groups : int
      The

    Returns
    -------
    permuted : int
      The
    """
    permuted = condition.copy()
    prng = RandomState(seed)

    for g in groups:
        gg = group == g
        permuted[gg] = prng.permutation(condition[gg])
    return permuted


def binom_conf_interval(n, x, cl=0.975, alternative="two-sided", p=None,
                        xtol=1e-12, rtol=4.4408920985006262e-16, maxiter=100):
    """
    Compute the confidence interval for a binomial.

    Parameters
    ----------
    n : int
      The number of Bernoulli trials.
    x : int
      The number of successes.
    cl : float in (0, 1)
      The desired confidence level.
    alternative : {"two-sided", "less", "greater"}
      Indicates the alternative hypothesis.
    p : float in (0, 1)
      The probability of success in each trial.
    xtol : float
      Tolerance
    rtol : float
      Tolerance
    maxiter : int
      Maximum number of iterations.

    Returns
    -------
    ci_low, ci_upp : float
        lower and upper confidence level with coverage (approximately)
        1-alpha.
    """
    if p is None:
        p = x / n
    ci_low = 0.0
    ci_upp = 1.0

    if alternative == 'both':
        cl = 1 - (1-cl)/2

    if alternative != "greater" and x > 0:
        f = lambda q: cl - binom.cdf(x - 1, n, q)
        ci_low = brentq(f, 0.0, p, xtol, rtol, maxiter)
    elif alternative != "less" and x < n:
        f = lambda q: binom.cdf(x, n, q) - (1 - cl)
        ci_upp = brentq(f, 1.0, p, xtol, rtol, maxiter)

    return ci_low, ci_upp


def permuTestMean(x, y, reps=10**5, stat='mean', alternative="greater",
                  CI=False, level=0.95, seed=None):
    """
    One-sided or two-sided, two-sample permutation test for equality of
    two means, with p-value estimated by simulated random sampling with
    reps replications.

    Tests the hypothesis that x and y are a random partition of x,y
    against the alternative that x comes from a population with mean
        (a) greater than that of the population from which y comes,
            if side = 'greater_than'
        (b) less than that of the population from which y comes,
            if side = 'less_than'
        (c) different from that of the population from which y comes,
            if side = 'both'

    If stat == 'mean', the test statistic is (mean(x) - mean(y))
    (equivalently, sum(x), since those are monotonically related)

    If stat == 't', the test statistic is the two-sample t-statistic--but
    the p-value is still estimated by the randomization, approximating
    the permutation distribution.
    The t-statistic is computed using scipy.stats.ttest_ind

    If CI == 'upper', computes an upper confidence bound on the true
    p-value based on the simulations by inverting Binomial tests.

    If CI == 'lower', computes a lower confidence bound on the true
    p-value based on the simulations by inverting Binomial tests.

    If CI == 'both', computes lower and upper confidence bounds on the true
    p-value based on the simulations by inverting Binomial tests.

    CL is the confidence limit for the confidence bounds.

    output is the estimated p-value and the test statistic, if CI == False
    output is <estimated p-value, confidence bound on p-value, test statistic>
         if CI in {'lower','upper'}
    output is <estimated p-value,
         [lower confidence bound, upper confidence bound], test statistic>,
          if CI == 'both'

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
    z = np.concatenate([x, y])   # pooled responses
    stats = {
        'mean': lambda u: np.mean(u[:len(x)]) - np.mean(u[len(x):]),
        't': lambda u: ttest_ind(
            u[:len(y)], u[len(y):], equal_var=True)[0]
    }
    tst = stats[stat]

    theStat = {
        'greater': tst,
        'less': lambda u: -tst(u),
        'two-sided': lambda u: math.fabs(tst(u))
    }

    ts = theStat[alternative](z)
    hits = np.sum([(theStat[alternative](prng.permutation(z)) >= ts)
                   for i in range(reps)])

    if CI in ["two-sided", "less", "greater"]:
        return hits / reps, binom_conf_interval(reps, level, alternative), ts
    else:
        return hits / reps, ts


def stratifiedPermutationTestMean(group, condition, response,
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
    if (len(groups) < 2):
        raise ValueError('Number of groups must be at least 2.')
    elif (len(groups) == 2):
        stat = lambda u: u[0] - u[1]
    elif (len(groups) > 2):
        stat = lambda u: np.std(u)
    for g in groups:
        gg = group == g
        x = [gg & (condition == c) for c in conditions]
        tst += stat([response[x[j]].mean() for j in range(len(x))])
    return tst


def stratifiedPermutationTest(group, condition, response, iterations=1.0e4,
                              testStatistic=stratifiedPermutationTestMean):
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
                                        group, condition, groups),
                                    response, groups, conditions
                                    )

        conds = [dist <= tst, dist >= tst, abs(dist) >= abs(tst)]
        pLeft, pRight, pBoth = [np.count_nonzero(c)/iterations for c in conds]
        return pLeft, pRight, pBoth, tst, dist
