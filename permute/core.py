"""
Core functions.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.stats import ttest_ind, ttest_1samp

from .utils import get_prng, potential_outcomes, binom_conf_interval
from .binomialp import binomial_p


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
    left_pv = np.sum(sims <= tst) / reps
    right_pv = np.sum(sims >= tst) / reps
    two_sided_pv = np.min([1, 2 * np.min([left_pv, right_pv])])
    return tst, left_pv, right_pv, two_sided_pv, sims


def two_sample_core(potential_outcomes_all, nx, tst_stat, alternative='greater',
                    reps=10**5, keep_dist=False, seed=None):
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

    rr = list(range(potential_outcomes_all.shape[0]))
    tst = tst_stat(potential_outcomes_all[:nx, 0],
                   potential_outcomes_all[nx:, 1])

    thePvalue = {
        'greater': lambda p: p,
        'less': lambda p: 1 - p,
        'two-sided': lambda p: 2 * np.min([p, 1 - p])
    }

    if keep_dist:
        dist = np.empty(reps)
        for i in range(reps):
            prng.shuffle(rr)
            pp = np.take(potential_outcomes_all, rr, axis=0)
            dist[i] = tst_stat(pp[:nx, 0], pp[nx:, 1])
        hits = np.sum(dist >= tst)
        return thePvalue[alternative](hits / reps), dist
    else:
        hits = 0
        for i in range(reps):
            prng.shuffle(rr)
            pp = np.take(potential_outcomes_all, rr, axis=0)
            hits += tst_stat(pp[:nx, 0], pp[nx:, 1]) >= tst
        return thePvalue[alternative](hits / reps)


def two_sample(x, y, reps=10**5, stat='mean', alternative="greater",
               keep_dist=False, seed=None):
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
            that function.  The function should take a permutation of the pooled
            data and compute the test function from it. For instance, if the
            test statistic is the Kolmogorov-Smirnov distance between the
            empirical distributions of the two samples, $\max_t |F_x(t) - F_y(t)|$,
            the test statistic could be written:

            f = lambda u: np.max( \
                [abs(sum(u[:len(x)]<=v)/len(x)-sum(u[len(x):]<=v)/len(y)) for v in u]\
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
                          reps=reps, keep_dist=keep_dist, seed=seed)

    if keep_dist:
        return res[0], observed_tst, res[1]
    else:
        return res, observed_tst


def two_sample_shift(x, y, reps=10**5, stat='mean', alternative="greater",
                     keep_dist=False, seed=None, shift=None):
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
            that function.  The function should take a permutation of the pooled
            data and compute the test function from it. For instance, if the
            test statistic is the Kolmogorov-Smirnov distance between the
            empirical distributions of the two samples, $\max_t |F_x(t) - F_y(t)|$,
            the test statistic could be written:

            f = lambda u: np.max( \
                [abs(sum(u[:len(x)]<=v)/len(x)-sum(u[len(x):]<=v)/len(y)) for v in u]\
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
                          reps=reps, keep_dist=keep_dist, seed=seed)

    if keep_dist:
        return res[0], observed_tst, res[1]
    else:
        return res, observed_tst


def two_sample_conf_int(x, y, cl=0.95, alternative="two-sided", seed=None,
                        reps=10**4, stat="mean", shift=None):
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
            that function.  The function should take a permutation of the pooled
            data and compute the test function from it. For instance, if the
            test statistic is the Kolmogorov-Smirnov distance between the
            empirical distributions of the two samples, $\max_t |F_x(t) - F_y(t)|$,
            the test statistic could be written:

            f = lambda u: np.max( \
                [abs(sum(u[:len(x)]<=v)/len(x)-sum(u[len(x):]<=v)/len(y)) for v in u]\
                )
    shift : float
        The relationship between x and y under the null hypothesis.

        (a) If None, the relationship is assumed to be additive (e.g. x = y+d)
        (b) A tuple containing the function and its inverse $(f, f^{-1})$, so
            $x_i = f(y_i, d)$ and $y_i = f^{-1}(x_i, d)$

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
                                                shift=q, reps=reps, stat=stat)[0]
        else:
            g = lambda q: cl - two_sample_shift(x, y, alternative="less", seed=seed,
                                                shift=(lambda u: f(u, q), lambda u: finverse(u, q)), reps=reps, stat=stat)[0]
        ci_low = brentq(g, -2 * shift_limit, 2 * shift_limit)

    if alternative != "lower":
        if shift is None:
            g = lambda q: cl - two_sample_shift(x, y, alternative="greater", seed=seed,
                                                shift=q, reps=reps, stat=stat)[0]
        else:
            g = lambda q: cl - two_sample_shift(x, y, alternative="greater", seed=seed,
                                                shift=(lambda u: f(u, q), lambda u: finverse(u, q)), reps=reps, stat=stat)[0]
        ci_upp = brentq(g, -2 * shift_limit, 2 * shift_limit)

    return ci_low, ci_upp


#ROUGH DRAFT: One sample test for percentile.
def one_sample_percentile(x, y=None, p=50, reps=10**5, alternative="greater", keep_dist=False, seed=None):
    r"""
    One-sided or two-sided test for the percentile P of a population distribution.
    assuming there is an P/100 chance for each value of the sample to be in the
    Pth percentile.

    The null hypothesis is that there is an equal P/100 chance for any value of the 
    sample to lie at or below the sample Pth percentile.

    This test defaults to P=50.
    
    Parameters
    ----------
    x : array-like
        Sample 1
    y : array-like
        Sample 2. Must preserve the order of pairs with x.
        If None, x is taken to be the one sample.
    p: int in [0,100]
        Percentile of interest to test.
    reps : int
        number of repetitions
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
        the test statistic: Number of values at or below percentile P of sample
    list
       distribution of test statistics (only if keep_dist == True)
    """

    if y is None:
        z = x
    elif len(x) != len(y):
        raise ValueError('x and y must be pairs')
    else:
        z = np.array(x) - np.array(y)

    if not 0 <= p <= 100:
        raise ValueError('p must be between 0 and 100')

    return binomial_p(np.sum(z <= np.percentile(z, p)), len(z), p/100, reps=reps, \
        alternative=alternative, keep_dist=keep_dist, seed=seed)



# ROUGH DRAFT: One sample confidence intervals for percentiles
def one_sample_percentile_conf_int(x, y=None, p=50, cl=0.95, alternative="two-sided", seed=None):
    """
    Confidence intervals for a percentile P of the population distribution of a
    univariate or paired sample.
    
    Compute a confidence interval for a binomial p, the probability of success in each trial.

    Parameters
    ----------
    n : int
        The number of Bernoulli trials.
    x : int
        The number of successes.
    cl : float in (0, 1)
        The desired confidence level.
    alternative : {"two-sided", "lower", "upper"}
        Indicates the alternative hypothesis.
    p : int in [0,100]
        Desired percentile of interest to test.
    kwargs : dict
        Key word arguments

    Returns
    -------
    tuple
        lower and upper confidence level with coverage (approximately)
        1-alpha.

    """

    if y is None:
        z = x
    elif len(x) != len(y):
        raise ValueError('x and y must be pairs')
    else:
        z = np.array(x) - np.array(y)

    if not 0 <= p <= 100:
        raise ValueError('p must be between 0 and 100')

    conf_int = binom_conf_interval(len(z), np.sum(z <= np.percentile(z, p)), cl=cl, alternative="two-sided", p=p/100)
    return (np.percentile(z, conf_int[0]*100), np.percentile(z, conf_int[1]*100))


def one_sample(x, y=None, reps=10**5, stat='mean', alternative="greater",
               keep_dist=False, seed=None):
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
        'greater': lambda p: p,
        'less': lambda p: 1 - p,
        'two-sided': lambda p: 2 * np.min([p, 1 - p])
    }

    stats = {
        'mean': lambda u: np.mean(u),
        't': lambda u: ttest_1samp(u, 0)[0],
        'median': lambda u: np.median(u)
    }

    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]

    tst = tst_fun(z)

    n = len(z)
    if keep_dist:
        dist = []
        for i in range(reps):
            dist.append(tst_fun(z * (1 - 2 * prng.binomial(1, .5, size=n))))
        hits = np.sum(dist >= tst)
        return thePvalue[alternative](hits / reps), tst, dist
    else:
        hits = np.sum([(tst_fun(z * (1 - 2 * prng.binomial(1, .5, size=n)))) >= tst
                       for i in range(reps)])
        return thePvalue[alternative](hits / reps), tst


def one_sample_shift(x, y=None, reps=10**5, stat='mean', alternative="greater",
               keep_dist=False, seed=None, shift = None):
    r"""
    One-sided or two-sided, one-sample permutation test for the mean,
    with p-value estimated by simulated random sampling with
    reps replications.

    Alternatively, a permutation test for equality of means of two paired
    samples.

    This function assumed a shift model. Given a shift (float), this test will
    find the 

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
    shift : float
        Assumption of symmetry around the shift, d.

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
        'greater': lambda p: p,
        'less': lambda p: 1 - p,
        'two-sided': lambda p: 2 * np.min([p, 1 - p])
    }
    stats = {
        'mean': lambda u: np.mean(u),
        't': lambda u: ttest_1samp(u, 0)[0],
        'median': lambda u: np.median(u)
    }
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]

    tst = tst_fun(z)

    if shift is None:
        shift = float(0)

    n = len(z)
    if keep_dist:
        dist = []
        for i in range(reps):
            dist.append(tst_fun(shift + (z - shift) * (1 - 2 * prng.binomial(1, .5, size=n))))
        hits = np.sum(dist >= tst)
        return thePvalue[alternative](hits / reps), tst, dist
    else:
        hits = np.sum([(tst_fun(shift + (z - shift) * (1 - 2 * prng.binomial(1, .5, size=n)))) >= tst
                       for i in range(reps)])
        return thePvalue[alternative](hits / reps), tst




def one_sample_conf_int(x, y = None, cl=0.95, alternative="two-sided", seed=None,
                        reps=10**4, stat="mean", shift=None):
    """
    One-sided or two-sided confidence interval for a test statistic of a sample with
    or paired sample.  The default is the two-sided confidence interval for the mean of a sample x.

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
            that function.  The function should take a permutation of the pooled
            data and compute the test function from it. For instance, if the
            test statistic is the Kolmogorov-Smirnov distance between the
            empirical distributions of the two samples, $\max_t |F_x(t) - F_y(t)|$,
            the test statistic could be written:

            f = lambda u: np.max( \
                [abs(sum(u[:len(x)]<=v)/len(x)-sum(u[len(x):]<=v)/len(y)) for v in u]\
                )
    shift : float
        The relationship between x and y under the null hypothesis.

        (a) If None, the relationship is assumed to be additive (e.g. x = y+d)
        (b) A tuple containing the function and its inverse $(f, f^{-1})$, so
            $x_i = f(y_i, d)$ and $y_i = f^{-1}(x_i, d)$

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

    if y is None:
        z = x
    elif len(x) != len(y):
        raise ValueError('x and y must be pairs')
    else:
        z = np.array(x) - np.array(y)

    if shift is None:
        shift_limit = max(z) - min(z)

    elif isinstance(shift, tuple):
        assert (callable(shift[0])), "Supply f and finverse in shift tuple"
        assert (callable(shift[1])), "Supply f and finverse in shift tuple"
        f = shift[0]
        finverse = shift[1]
        # Check that f is increasing in d; this is very ad hoc!
        assert (f(5, 1) < f(5, 2)), "f must be increasing in the parameter d"

        shift_limit = max(abs(fsolve(lambda d: f(max(z), d) - min(z), 0)),
                          abs(fsolve(lambda d: f(min(z), d) - max(z), 0)))
        # FIXME: unused observed
        # observed = fsolve(lambda d: np.mean(x) - np.mean(f(y, d)), 0)
    else:
        raise ValueError("Bad input for shift")

    stats = {
        'mean': lambda u: np.mean(u),
        't': lambda u: ttest_1samp(u, 0)[0],
        'median': lambda u: np.median(u)
    }
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]

    tst = tst_fun(z)

    ci_low = tst - shift_limit
    ci_upp = tst + shift_limit

    if alternative == 'two-sided':
        cl = 1 - (1 - cl) / 2

    if alternative != "upper":
        # if shift is None:
        #     g = lambda q: cl - one_sample_shift(x, y, alternative="less", seed=seed, reps=reps, stat=stat, shift=q)[0]
        # else:
        g = lambda q: cl - one_sample_shift(z, alternative="less", seed=seed, \
            reps=reps, stat=stat, shift=q)[0]
        ci_low = brentq(g, tst - 2 * shift_limit, tst + 2 * shift_limit)

    if alternative != "lower":
        # if shift is None:
        #     g = lambda q: cl - one_sample_shift(x, y, alternative="greater", seed=seed, reps=reps, stat=stat, shift=q)[0]
        # else:
        g = lambda q: cl - one_sample_shift(z, alternative="greater", seed=seed, \
            reps=reps, stat=stat, shift=q)[0]
        ci_upp = brentq(g, tst - 2 * shift_limit, tst + 2 * shift_limit)

    return ci_low, ci_upp


    
