# -*- coding: utf-8 -*-

"""
Tests for the slope in linear regression
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.stats import linregress

from .utils import get_prng, permute


def compute_ts(x, y):
    """
    Compute the test statistic

    Parameters
    ----------
    x, y  : array_like
        Input data

    Returns
    -------
    ts : float
        the least squares estimate of the slope normalized by its standard error
    """
    slope, intercepts, rval, pval, stderr = linregress(x, y)
    return slope / stderr


def simulate_ts_dist(x, y, obs_ts=None, num_perm=10000,
                     keep_dist=False, seed=None):
    """
    Simulates the permutation distribution of the test statistic for
    a matrix of ratings ``ratings``

    If ``obs_ts`` is not ``None``, computes the reference value of the test
    statistic before the first permutation. Otherwise, uses the value
    ``obs_ts`` for comparison.

    If ``keep_dist``, return the distribution of values of the test statistic;
    otherwise, return only the number of permutations for which the value of
    the test statistic is at least as large as ``obs_ts``.

    Parameters
    ----------
    ratings : array_like
              Input array of dimension [R, Ns]
    obs_ts : float
             if None, ``obs_ts`` is calculated as the value of the test
             statistic for the original data
    num_perm : int
           number of random permutation of the elements of each row of ratings
    keep_dist : bool
                flag for whether to store and return the array of values of
                the test statistic
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator

    Returns
    -------
    dict
        A dictionary containing:

        obs_ts : int
            observed value of the test statistic for the input data, or
            the input value of ``obs_ts`` if ``obs_ts`` was given as input
        geq : int
            number of iterations for which the test statistic was greater
            than or equal to ``obs_ts``
        num_perm : int
            number of permutations
        pvalue : float
            geq / num_perm
        dist : array-like
            if ``keep_dist``, the array of values of the test statistic
            from the ``num_perm`` iterations.  Otherwise, ``None``.
    """
    y1 = y.copy()
    prng = get_prng(seed)

    if obs_ts is None:
        obs_ts = compute_ts(x, y1)

    if keep_dist:
        dist = np.zeros(num_perm)
        for i in range(num_perm):
            permute(y1, prng)
            dist[i] = compute_ts(x, y1)
        geq = np.sum(dist >= obs_ts)
    else:
        dist = None
        geq = 0
        for i in range(num_perm):
            permute(y1, prng)
            geq += (compute_ts(x, y1) >= obs_ts)
    return {"obs_ts": obs_ts, "geq": geq, "num_perm": num_perm,
            "pvalue": geq/num_perm, "dist": dist}

