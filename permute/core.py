# -*- coding: utf-8 -*-

"""
Core.
"""

from __future__ import division, print_function, absolute_import

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import (binom, ttest_ind, ttest_1samp)

from .utils import get_prng, binom_conf_interval


def corr(x, y, reps=10**4, seed=None):
    """
    Simulate permutation p-value for Spearman correlation coefficient

    Parameters
    ----------
    x : array-like
    y : array-like
    reps : int
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator


    Returns
    -------
    tuple
        Returns test statistic, left-sided p-value,
        right-sided p-value, two-sided p-value, simulated distribution
    """
    prng = get_prng(seed)
    tst = np.corrcoef(x, y)[0, 1]
    sims = [np.corrcoef(prng.permutation(x), y)[0, 1] for i in range(reps)]
    left_pv = np.sum(sims <= tst)/reps
    right_pv = np.sum(sims >= tst)/reps
    two_sided_pv = np.sum(np.abs(sims) >= np.abs(tst))/reps
    return tst, left_pv, right_pv, two_sided_pv, sims


def two_sample(x, y, reps=10**5, stat='mean', alternative="greater",
               keep_dist=False, interval=False, level=0.95, seed=None):
    """
    One-sided or two-sided, two-sample permutation test for equality of
    two means, with p-value estimated by simulated random sampling with
    reps replications.

    Tests the hypothesis that x and y are a random partition of x,y
    against the alternative that x comes from a population with mean

    (a) greater than that of the population from which y comes,
        if side = 'greater'
    (b) less than that of the population from which y comes,
        if side = 'less'
    (c) different from that of the population from which y comes,
        if side = 'two-sided'

    If ``keep_dist``, return the distribution of values of the test statistic;
    otherwise, return only the number of permutations for which the value of
    the test statistic and p-value.

    Parameters
    ----------
    x : array-like
        Sample 1
    y : array-like
        Sample 2
    reps : int
        number of repetitions
    stat : {'mean', 't'}
        The test statistic.

        (a) If stat == 'mean', the test statistic is (mean(x) - mean(y))
            (equivalently, sum(x), since those are monotonically related)
        (b) If stat == 't', the test statistic is the two-sample t-statistic--
            but the p-value is still estimated by the randomization,
            approximating the permutation distribution.
            The t-statistic is computed using scipy.stats.ttest_ind
        (c) If stat is a function (a callable object), the test statistic is
            that function.  The function should take a permutation of the pooled
            data and compute the test function from it. For instance, if the
            test statistic is the Kolmogorov-Smirnov distance between the
            empirical distributions of the two samples, max_t |F_x(t) - F_y(t)|,
            the test statistic could be written:

            f = lambda u: np.max( \
                [abs(sum(u[:len(x)]<=v)/len(x)-sum(u[len(x):]<=v)/len(y)) for v in u]\
                )

    alternative : {'greater', 'less', 'two-sided'}
        The alternative hypothesis to test
    keep_dist : bool
        flag for whether to store and return the array of values
        of the irr test statistic
    interval : {'upper', 'lower', 'two-sided'}
        The type of confidence interval

        (a) If interval == 'upper', computes an upper confidence bound on the
            true p-value based on the simulations by inverting Binomial tests.
        (b) If interval == 'lower', computes a lower confidence bound on the
            true p-value based on the simulations by inverting Binomial tests.
        (c) If interval == 'two-sided', computes lower and upper confidence
            bounds on the true p-value based on the simulations by inverting
            Binomial tests.
    level : float in (0, 1)
        the confidence limit for the confidence bounds.
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
        the test statistic
    tuple
        These values are only returned if `level` == True

        (a) confidence bound on p-value,
            if interval in {'lower','upper'}
        (b) [lower confidence bound, upper confidence bound],
            if interval == 'two-sided'
    list
        The distribution of test statistics.
        These values are only returned if `keep_dist` == True
    """
    prng = get_prng(seed)
    z = np.concatenate([x, y])   # pooled responses

    # If stat is callable, use it as the test function. Otherwise, look in the dictionary

    stats = {
        'mean': lambda u: np.mean(u[:len(x)]) - np.mean(u[len(x):]),
        't': lambda u: ttest_ind(
            u[:len(y)], u[len(y):], equal_var=True)[0]
    }
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]

    theStat = {
        'greater': tst_fun,
        'less': lambda u: -tst_fun(u),
        'two-sided': lambda u: math.fabs(tst_fun(u))
    }

    tst = theStat[alternative](z)
    if keep_dist:
        dist = np.empty(reps)
        for i in range(reps):
            dist[i] = theStat[alternative](prng.permutation(z))
        hits = np.sum(dist >= tst)
        if interval in ["upper", "lower", "two-sided"]:
            return (hits/reps, tst,
                    binom_conf_interval(reps, hits, level, interval), dist)
        else:
            return hits/reps, tst, dist
    else:
        hits = np.sum([(theStat[alternative](prng.permutation(z)) >= tst)
                       for i in range(reps)])

    if interval in ["upper", "lower", "two-sided"]:
        return (hits/reps, tst,
                binom_conf_interval(reps, hits, level, interval))
    else:
        return hits/reps, tst


def one_sample(x, y=None, reps=10**5, stat='mean', alternative="greater",
               keep_dist=False, seed=None):
    """
    One-sided or two-sided, one-sample permutation test for the mean,
    with p-value estimated by simulated random sampling with
    reps replications.

    Alternatively, a permutation test for equality of means of two paired
    samples.

    Tests the hypothesis that x is distributed symmetrically symmetric about 0
    (or x and y have the same center) against the alternative that x comes from
    a population with mean

    (a) greater than 0 (greater than that of the population from which y comes),
        if side = 'greater'
    (b) less than 0 (less than that of the population from which y comes),
        if side = 'less'
    (c) different from 0 (different from that of the population from which y comes),
        if side = 'two-sided'

    If ``keep_dist``, return the distribution of values of the test statistic;
    otherwise, return only the number of permutations for which the value of
    the test statistic and p-value.

    Parameters
    ----------
    x : array-like
        Sample 1
    y : array-like
        Sample 2. Must preserve the order of pairs with x.
        If None, x is taken to be the one sample.
    reps : int
        number of repetitions
    stat : {'mean', 't'}
        The test statistic.

        (a) If stat == 'mean', the test statistic is (mean(x) - mean(y))
            (equivalently, sum(x), since those are monotonically related)
        (b) If stat == 't', the test statistic is the two-sample t-statistic--
            but the p-value is still estimated by the randomization,
            approximating the permutation distribution.
            The t-statistic is computed using scipy.stats.ttest_ind
        (c) FIXME: Explanation or example of how to pass in a function,
            instead of a str
    alternative : {'greater', 'less', 'two-sided'}
        The alternative hypothesis to test
    keep_dist : bool
        flag for whether to store and return the array of values
        of the irr test statistic
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
        the test statistic
    list
        The distribution of test statistics.
        These values are only returned if `keep_dist` == True
    """
    prng = get_prng(seed)

    if y is None:
        z = x
    elif len(x)!=len(y):
        raise ValueError('x and y must be pairs')
    else:
        z = np.array(x)-np.array(y)

    # FIXME: Type check: we may want to pass in a function for argument 'stat'
    # FIXME: If function, use that. Otherwise, look in the dictionary
    stats = {
        'mean': lambda u: np.mean(u),
        't': lambda u: ttest_1samp(u, 0)[0]
    }
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]

    theStat = {
        'greater': tst_fun,
        'less': lambda u: -tst_fun(u),
        'two-sided': lambda u: math.fabs(tst_fun(u))
    }

    tst = theStat[alternative](z)
    n = len(z)
    if keep_dist:
        dist = []
        for i in range(reps):
            dist.append(theStat[alternative](z*(1-2*prng.binomial(1,.5,size=n))))
        hits = np.sum(dist >= tst)
        return hits/reps, tst, dist
    else:
        hits = np.sum([(theStat[alternative](z*(1-2*prng.binomial(1,.5,size=n)))) >= tst
                       for i in range(reps)])
        return hits/reps, tst