"""
Core functions.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.stats import ttest_ind, ttest_1samp
from fractions import Fraction

from .utils import get_prng, potential_outcomes, permute


def corr(x, y, alternative='greater', reps=10**4, seed=None, plus1=True):
    r"""
    Simulate permutation p-value for Pearson correlation coefficient

    Parameters
    ----------
    x : array-like
    y : array-like
    alternative : {'greater', 'less', 'two-sided'}
        The alternative hypothesis to test
    reps : int
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
    tuple
        Returns test statistic, p-value, simulated distribution
    """
    prng = get_prng(seed)
    tst = np.corrcoef(x, y)[0, 1]
    sims = [np.corrcoef(permute(x, prng), y)[0, 1] for i in range(reps)]
    left_pv = (np.sum(sims <= tst)+plus1) / (reps+plus1)
    right_pv = (np.sum(sims >= tst)+plus1) / (reps+plus1)
    if alternative == 'greater':
        pvalue = right_pv
    elif alternative == 'less':
        pvalue = left_pv
    elif alternative == 'two-sided':
        pvalue = np.min([1, 2 * np.min([left_pv, right_pv])])
    return tst, pvalue, sims


def spearman_corr(x, y, alternative='greater', reps=10**4, seed=None, plus1=True):
    r"""
    Simulate permutation p-value for Spearman correlation coefficient

    Parameters
    ----------
    x : array-like
    y : array-like
    alternative : {'greater', 'less', 'two-sided'}
        The alternative hypothesis to test
    reps : int
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
    tuple
        Returns test statistic, p-value, simulated distribution
    """
    
    xnew = np.argsort(x)+1
    ynew = np.argsort(y)+1
    return corr(xnew, ynew, alternative=alternative, reps=reps, seed=seed)


def two_sample_core(potential_outcomes_all, nx, tst_stat, alternative='greater',
                    reps=10**5, keep_dist=False, seed=None, plus1=True):
    r"""
    Main workhorse function for two_sample and two_sample_shift

    Parameters
    ----------
    potential_outcomes_all : array-like
        2D array of potential outcomes under treatment (1st column)
        and control (2nd column). To be passed in from potential_outcomes
    nx : int
        Size of the treatment group x
    reps : int
        number of repetitions
    tst_stat: function
        The test statistic
    alternative : {'greater', 'less', 'two-sided'}
        The alternative hypothesis to test
    keep_dist : bool
        flag for whether to store and return the array of values
        of the test statistic. Default is False.
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

    rr = list(range(potential_outcomes_all.shape[0]))
    tst = tst_stat(potential_outcomes_all[:nx, 0],
                   potential_outcomes_all[nx:, 1])

    thePvalue = {
        'greater': lambda p: p+plus1/(reps+plus1),
        'less': lambda p: 1 - p,
        'two-sided': lambda p: 2 * np.min([0.5, \
                                    p+plus1/(reps+plus1), \
                                    1 - p])
    }

    if keep_dist:
        dist = np.empty(reps)
        for i in range(reps):
            prng.shuffle(rr)
            pp = np.take(potential_outcomes_all, rr, axis=0)
            dist[i] = tst_stat(pp[:nx, 0], pp[nx:, 1])
        hits = np.sum(dist >= tst)
        return thePvalue[alternative](hits / (reps+plus1)), dist
    else:
        hits = 0
        for i in range(reps):
            prng.shuffle(rr)
            pp = np.take(potential_outcomes_all, rr, axis=0)
            hits += tst_stat(pp[:nx, 0], pp[nx:, 1]) >= tst
        return thePvalue[alternative](hits / (reps+plus1))


def two_sample(x, y, reps=10**5, stat='mean', alternative="greater",
               keep_dist=False, seed=None, plus1=True):
    r"""
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
            that function. The function should take two arguments:
            given a permutation of the pooled data, the first argument is the
            "new" x and the second argument is the "new" y.
            For instance, if the test statistic is the Kolmogorov-Smirnov distance 
            between the empirical distributions of the two samples, 
            $\max_t |F_x(t) - F_y(t)|$, the test statistic could be written:

            f = lambda u, v: np.max( \
                [abs(sum(u<=val)/len(u)-sum(v<=val)/len(v)) for val in np.concatenate([u, v])]\
                )

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
    # Set up potential outcomes; under the null, all units are exchangeable
    pot_out_all = np.column_stack(
        [np.concatenate([x, y]), np.concatenate([x, y])])

    # If stat is callable, use it as the test function. Otherwise, look in the
    # dictionary
    stats = {
        'mean': lambda u, v: np.mean(u) - np.mean(v),
        't': lambda u, v: ttest_ind(u, v, equal_var=True)[0]
    }
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]

    nx = len(x)
    observed_tst = tst_fun(pot_out_all[:nx, 0], pot_out_all[nx:, 1])

    res = two_sample_core(pot_out_all, nx, tst_fun, alternative=alternative,
                          reps=reps, keep_dist=keep_dist, seed=seed, plus1=plus1)

    if keep_dist:
        return res[0], observed_tst, res[1]
    else:
        return res, observed_tst


def two_sample_shift(x, y, reps=10**5, stat='mean', alternative="greater",
                     keep_dist=False, seed=None, shift=None, plus1=True):
    r"""
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
            that function. The function should take two arguments:
            given a permutation of the pooled data, the first argument is the
            "new" x and the second argument is the "new" y.
            For instance, if the test statistic is the Kolmogorov-Smirnov distance 
            between the empirical distributions of the two samples, 
            $\max_t |F_x(t) - F_y(t)|$, the test statistic could be written:

            f = lambda u, v: np.max( \
                [abs(sum(u<=val)/len(u)-sum(v<=val)/len(v)) for val in np.concatenate([u, v])]\
                )
                
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
    shift : float
        The relationship between x and y under the null hypothesis.

        (a) A constant scalar shift in the distribution of y. That is, x is equal
            in distribution to y + shift.
        (b) A tuple containing the function and its inverse $(f, f^{-1})$, so
            $x_i = f(y_i)$ and $y_i = f^{-1}(x_i)$
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
    # Set up potential outcomes according to shift
    if isinstance(shift, float) or isinstance(shift, int):
        # Potential outcomes for all units under treatment
        pot_outx = np.concatenate([x, y + shift])
        # Potential outcomes for all units under control
        pot_outy = np.concatenate([x - shift, y])
        pot_out_all = np.column_stack([pot_outx, pot_outy])
    elif isinstance(shift, tuple):
        assert (callable(shift[0])), "Supply f and finverse in shift tuple"
        assert (callable(shift[1])), "Supply f and finverse in shift tuple"
        pot_out_all = potential_outcomes(x, y, shift[0], shift[1])
    else:
        raise ValueError("Bad input for shift")

    # If stat is callable, use it as the test function. Otherwise, look in the
    # dictionary
    stats = {
        'mean': lambda u, v: np.mean(u) - np.mean(v),
        't': lambda u, v: ttest_ind(u, v, equal_var=True)[0]
    }
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]

    nx = len(x)
    observed_tst = tst_fun(pot_out_all[:nx, 0], pot_out_all[nx:, 1])

    res = two_sample_core(pot_out_all, nx, tst_fun, alternative=alternative,
                          reps=reps, keep_dist=keep_dist, seed=seed, plus1=plus1)

    if keep_dist:
        return res[0], observed_tst, res[1]
    else:
        return res, observed_tst


def two_sample_conf_int(x, y, cl=0.95, alternative="two-sided", seed=None,
                        reps=10**4, stat="mean", shift=None, plus1=True):
    r"""
    One-sided or two-sided confidence interval for the parameter determining
    the treatment effect.  The default is the "shift model", where we are
    interested in the parameter d such that x is equal in distribution to
    y + d. In general, if we have some family of invertible functions parameterized
    by d, we'd like to find d such that x is equal in distribution to f(y, d).

    Parameters
    ----------
    x : array-like
        Sample 1
    y : array-like
        Sample 2
    cl : float in (0, 1)
        The desired confidence level. Default 0.95.
    alternative : {"two-sided", "lower", "upper"}
        Indicates the alternative hypothesis.
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator
    reps : int
        number of repetitions in two_sample
    stat : {'mean', 't'}
        The test statistic.

        (a) If stat == 'mean', the test statistic is (mean(x) - mean(y))
            (equivalently, sum(x), since those are monotonically related)
        (b) If stat == 't', the test statistic is the two-sample t-statistic--
            but the p-value is still estimated by the randomization,
            approximating the permutation distribution.
            The t-statistic is computed using scipy.stats.ttest_ind
        (c) If stat is a function (a callable object), the test statistic is
            that function. The function should take two arguments:
            given a permutation of the pooled data, the first argument is the
            "new" x and the second argument is the "new" y.
            For instance, if the test statistic is the Kolmogorov-Smirnov distance 
            between the empirical distributions of the two samples, 
            $\max_t |F_x(t) - F_y(t)|$, the test statistic could be written:

            f = lambda u, v: np.max( \
                [abs(sum(u<=val)/len(u)-sum(v<=val)/len(v)) for val in np.concatenate([u, v])]\
                )
                
    shift : float
        The relationship between x and y under the null hypothesis.

        (a) If None, the relationship is assumed to be additive (e.g. x = y+d)
        (b) A tuple containing the function and its inverse $(f, f^{-1})$, so
            $x_i = f(y_i, d)$ and $y_i = f^{-1}(x_i, d)$
    plus1 : bool
        flag for whether to add 1 to the numerator and denominator of the
        p-value based on the empirical permutation distribution. 
        Default is True.

    Returns
    -------
    tuple
        the estimated confidence limits

    Notes
    -----
    xtol : float
        Tolerance in brentq
    rtol : float
        Tolerance in brentq
    maxiter : int
        Maximum number of iterations in brentq
    """

    assert alternative in ("two-sided", "lower", "upper")

    if shift is None:
        shift_limit = max(abs(max(x) - min(y)), abs(max(y) - min(x)))
        # FIXME: unused observed
        # observed = np.mean(x) - np.mean(y)
    elif isinstance(shift, tuple):
        assert (callable(shift[0])), "Supply f and finverse in shift tuple"
        assert (callable(shift[1])), "Supply f and finverse in shift tuple"
        f = shift[0]
        finverse = shift[1]
        # Check that f is increasing in d; this is very ad hoc!
        assert (f(5, 1) < f(5, 2)), "f must be increasing in the parameter d"
        shift_limit = max(abs(fsolve(lambda d: f(max(y), d) - min(x), 0)),
                          abs(fsolve(lambda d: f(min(y), d) - max(x), 0)))
        # FIXME: unused observed
        # observed = fsolve(lambda d: np.mean(x) - np.mean(f(y, d)), 0)
    else:
        raise ValueError("Bad input for shift")
    ci_low = -shift_limit
    ci_upp = shift_limit

    if alternative == 'two-sided':
        cl = 1 - (1 - cl) / 2

    if alternative != "upper":
        if shift is None:
            g = lambda q: cl - two_sample_shift(x, y, alternative="less", seed=seed,
                                                shift=q, reps=reps, stat=stat, plus1=plus1)[0]
        else:
            g = lambda q: cl - two_sample_shift(x, y, alternative="less", seed=seed,
                                                shift=(lambda u: f(u, q), lambda u: finverse(u, q)), 
                                                reps=reps, stat=stat, plus1=plus1)[0]
        ci_low = brentq(g, -2 * shift_limit, 2 * shift_limit)

    if alternative != "lower":
        if shift is None:
            g = lambda q: cl - two_sample_shift(x, y, alternative="greater", seed=seed,
                                                shift=q, reps=reps, stat=stat, plus1=plus1)[0]
        else:
            g = lambda q: cl - two_sample_shift(x, y, alternative="greater", seed=seed,
                                                shift=(lambda u: f(u, q), lambda u: finverse(u, q)), 
                                                reps=reps, stat=stat, plus1=plus1)[0]
        ci_upp = brentq(g, -2 * shift_limit, 2 * shift_limit)

    return ci_low, ci_upp


def one_sample(x, y=None, reps=10**5, stat='mean', alternative="greater",
               keep_dist=False, seed=None, plus1=True):
    r"""
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
        The test statistic. The statistic is computed based on either z = x or
        z = x - y, if y is specified.

        (a) If stat == 'mean', the test statistic is mean(z).
        (b) If stat == 't', the test statistic is the t-statistic--
            but the p-value is still estimated by the randomization,
            approximating the permutation distribution.
        (c) If stat is a function (a callable object), the test statistic is
            that function.  The function should take a permutation of the
            data and compute the test function from it. For instance, if the
            test statistic is the maximum absolute value, $\max_i |z_i|$,
            the test statistic could be written:

            f = lambda u: np.max(abs(u))
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

    if y is None:
        z = x
    elif len(x) != len(y):
        raise ValueError('x and y must be pairs')
    else:
        z = np.array(x) - np.array(y)

    thePvalue = {
        'greater': lambda p: p+plus1/(reps+plus1),
        'less': lambda p: 1 - p,
        'two-sided': lambda p: 2 * np.min([0.5, \
                                    p+plus1/(reps+plus1), \
                                    1 - p])
    }
    stats = {
        'mean': lambda u: np.mean(u),
        't': lambda u: ttest_1samp(u, 0)[0]
    }
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]

    tst = tst_fun(z)
    n = len(z)
    if keep_dist:
        dist = np.empty(reps)
        for i in range(reps):
            dist[i] = tst_fun(z * (1 - 2 * prng.randint(0, 2, n)))
        hits = np.sum(dist >= tst)
        return thePvalue[alternative](hits / (reps+plus1)), tst, dist
    else:
        hits = np.sum([(tst_fun(z * (1 - 2 * prng.randint(0, 2, n)))) >= tst
                       for i in range(reps)])
        return thePvalue[alternative](hits / (reps+plus1)), tst
