# -*- coding: utf-8 -*-

"""
Core.
"""

from __future__ import division, print_function, absolute_import

import math

import numpy as np
from numpy.random import RandomState

from scipy.optimize import brentq

from scipy.stats import (binom, ttest_ind)


def permute_within_groups(group, condition, groups, prng=None):
    """
    Permutation of condition within each group.

    Parameters
    ----------
    group : array-like
      A 1-d array indicating group membership
    condition : array-like
      A 1-d array indcating treatment.
    groups : array-like
      The unique elements of group

    Returns
    -------
    permuted : array-like
      The within group permutation of condition.
    """
    permuted = condition.copy()
    if prng is None:
        prng = RandomState()

    # FIXME: do we need to pass `groups` in?
    # Yes, don't want to repeatedly identify unique elements
    # (avoid additional flops) -- maybe memoize
    for g in groups:
        gg = group == g
        # FIXME: Shuffle in place doesn't seem to work for slices
        # prng.shuffle(permuted[gg])
        permuted[gg] = prng.permutation(permuted[gg])
    return permuted


def permute_rows(m, prng=None):
    """
    Permute the rows of a matrix in-place

    Parameters
    ----------
    m : array-like
      A 2-d array
    prng : RandomState object or None
      The Pseudo-random number generator (used for replicability)

    Returns
    -------
    None
      Original matrix is permute in-place, nothing returned
    """
    if prng is None:
        prng = RandomState()

    for row in m:
        prng.shuffle(row)


def binom_conf_interval(n, x, cl=0.975, alternative="two-sided", p=None,
                        **kwargs):
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
    **kwargs : dict
      Key word arguments

    Returns
    -------
    ci_low, ci_upp : float
        lower and upper confidence level with coverage (approximately)
        1-alpha.

    Notes
    -----
    xtol : float
      Tolerance
    rtol : float
      Tolerance
    maxiter : int
      Maximum number of iterations.
    """
    if p is None:
        p = x / n
    ci_low = 0.0
    ci_upp = 1.0

    if alternative == 'two-sided':
        cl = 1 - (1-cl)/2

    # FIXME: should I check that alternative is valid?
    if alternative != "greater" and x > 0:
        f = lambda q: cl - binom.cdf(x - 1, n, q)
        ci_low = brentq(f, 0.0, p, *kwargs)
    if alternative != "less" and x < n:
        f = lambda q: binom.cdf(x, n, q) - (1 - cl)
        ci_upp = brentq(f, 1.0, p, *kwargs)

    return ci_low, ci_upp


def two_sample(x, y, reps=10**5, stat='mean', alternative="greater",
               interval=False, level=0.95, seed=None):
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

    Parameters
    ----------
    x : array-like
      Sample 1
    y : array-like
      Sample 2
    reps : int
      number of repetitions
    stat : {'mean', 't'}
      If stat == 'mean', the test statistic is (mean(x) - mean(y))
      (equivalently, sum(x), since those are monotonically related)

      If stat == 't', the test statistic is the two-sample t-statistic--but
      the p-value is still estimated by the randomization, approximating
      the permutation distribution.
      The t-statistic is computed using scipy.stats.ttest_ind
      # FIXME: Explanation or example of how to pass in a function, instead of a str
    interval : {'upper', 'lower', 'two-sided'}
      If interval == 'upper', computes an upper confidence bound on the true
      p-value based on the simulations by inverting Binomial tests.

      If interval == 'lower', computes a lower confidence bound on the true
      p-value based on the simulations by inverting Binomial tests.

      If interval == 'two-sided', computes lower and upper confidence bounds on
      the true p-value based on the simulations by inverting Binomial tests.
    level : float in (0, 1)
      the confidence limit for the confidence bounds.


    Returns
    -------
    output : int
      output is the estimated p-value and the test statistic, if level == False

      output is <estimated p-value, confidence bound on p-value, test statistic>
      if interval in {'lower','upper'}

      output is <estimated p-value,
      [lower confidence bound, upper confidence bound], test statistic>,
      if interval == 'two-sided'
    """
    prng = RandomState(seed)
    z = np.concatenate([x, y])   # pooled responses
    # FIXME: Type check: we may want to pass in a function for argument 'stat'
    # FIXME: If function, use that. Otherwise, look in the dictionary
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

    if interval in ["upper", "lower", "two-sided"]:
        return hits / reps, binom_conf_interval(n = reps, x = hits, cl = level, alternative = alternative), ts
    else:
        return hits / reps, ts
