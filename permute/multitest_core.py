# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:38:11 2022

@author: Clayton
"""


import numpy as np
from scipy.stats import ttest_ind, ttest_1samp

from .utils import get_prng, permute



""" TODO for multi testing core package

two_sample_conf_int once it is finalized

"""

def multitest_corr(x, y, alternative='greater', reps=10**4, seed=None, plus1=True):
    r"""
    Simulate permutation p-value for multiple Pearson correlation coefficients

    Parameters
    ----------
    x : array-like with shape (observations,tests)
    y : array-like with shape (observations,tests)
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
    # make sure the two samples have the same shape
    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape')
    # get number of hypotheses being tested
    num_tests = x.shape[1]
    prng = get_prng(seed)
    # create mask to grab the corr values we care about from np.corrcoef function (majority of what it returns we aren't interested in)
    corr_mat_mask = np.zeros((2*num_tests,2*num_tests),dtype=bool)
    corr_mat_mask[x.shape[1]+np.arange(num_tests),np.arange(num_tests)] = True
    # calculate correlations and grab the pairs we are interested in
    tst = np.corrcoef(x, y,rowvar=False)[corr_mat_mask]
    # permute to get null distribution
    sims = [np.corrcoef(permute(x, prng), y,rowvar=False)[corr_mat_mask] for i in range(reps)]
    # get percentiles of statistic
    left_pv = (np.sum(sims <= tst,axis=0)+plus1) / (reps+plus1)
    right_pv = (np.sum(sims >= tst,axis=0)+plus1) / (reps+plus1)
    # assign pvalue based on hypothesis
    if alternative == 'greater':
        pvalue = right_pv
    elif alternative == 'less':
        pvalue = left_pv
    elif alternative == 'two-sided':
        pvalue = np.min([1*np.ones(num_tests), 2 * np.min([left_pv, right_pv],axis=0)],axis=0)
    return tst, pvalue, sims

def multitest_spearman_corr(x, y, alternative='greater', reps=10**4, seed=None, plus1=True):
    r"""
    Simulate permutation p-value for multiple Spearman correlation coefficients

    Parameters
    ----------
    x : array-like with shape (observations,tests)
    y : array-like with shape (observations,tests)
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
    # sort observations per each test
    xnew = np.argsort(x,0)+1
    ynew = np.argsort(y,0)+1
    return multitest_corr(xnew, ynew, alternative=alternative, reps=reps, seed=seed)



def multitest_two_sample_core(potential_outcomes_all, nx, tst_stat, alternative='greater',
                    reps=10**5, keep_dist=False, seed=None, plus1=True, max_correct=False):
    r"""
    Main workhorse function for two_sample and two_sample_shift

    Parameters
    ----------
    potential_outcomes_all : array-like
        3D array [observations,tests,conditions] of multiple potential
        outcomes under treatment [:,:,0] and control [:,:,1]. 
        To be passed in from potential_outcomes
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
    max_correct : bool
        flag for whether to perform max statistic multiple testing
        correction. Builds the null distribution from the most extreme value
        across tests for each iteration of the permutation. Default is False.

    Returns
    -------
    float
        the estimated p-value
    list
        The distribution of test statistics.
        These values are only returned if `keep_dist` == True
    """
    prng = get_prng(seed)
    # get number of hypotheses being tested
    num_tests = potential_outcomes_all.shape[1]
     # create indexing vector
    rr = list(range(potential_outcomes_all.shape[0]))
    # get observed statistic
    tst = tst_stat(potential_outcomes_all[:nx,:,0], 
                   potential_outcomes_all[nx:,:, 1])
    # create dictionary to store functions for calculating p values
    thePvalue = {
        'greater': lambda pUp, pDn: pUp+plus1/(reps+plus1),
        'less': lambda pUp, pDn: pDn+plus1/(reps+plus1),
        'two-sided': lambda pUp, pDn: 2 * np.min(np.stack([0.5*np.ones(num_tests), \
                                    pUp+plus1/(reps+plus1), \
                                    pDn+plus1/(reps+plus1)],1),1)
    }
    # account for all combinations of keep_dist (keep distribution)
    # and max_correct (create max distribution to correct for multiple 
    # hypothesis testing) 
    if keep_dist:
        if max_correct:
            dist = np.empty(reps)
            for i in range(reps):
                # permute indexing vector
                prng.shuffle(rr)
                # grab shuffled values
                pp = np.take(potential_outcomes_all, rr, axis=0)
                # calculate statistic
                curr_tst = tst_stat(pp[:nx, :, 0], pp[nx:,: , 1])
                # grab the most extreme statistic observed across all tests
                dist[i] = max(curr_tst.min(), curr_tst.max(), key=abs)
            # calculate percentile 
            pUp = np.empty(num_tests)
            pDn = np.empty(num_tests)
            for i in range(num_tests):
                pUp[i] = np.sum(dist >= tst[i])/(reps+plus1)
                pDn[i] = np.sum(dist <= tst[i])/(reps+plus1)
            return thePvalue[alternative](pUp, pDn), dist
        else:
            dist = np.empty((reps,num_tests))
            for i in range(reps):
                # permute indexing vector
                prng.shuffle(rr)
                # grab shuffled values
                pp = np.take(potential_outcomes_all, rr, axis=0)
                # calculate statistic
                dist[i,:] = tst_stat(pp[:nx, :, 0], pp[nx:,: , 1])
            # calculate percentile
            pUp = np.sum(dist >= tst,axis=0)/(reps+plus1)
            pDn = np.sum(dist <= tst,axis=0)/(reps+plus1)
            return thePvalue[alternative](pUp, pDn), dist
    else:
        # preallocate vectors to store number of times our observed statistic
        # is greater than the permuted statistic(s). Keeps memory requirement low
        hitsUp = np.zeros(num_tests)
        hitsDn = np.zeros(num_tests)
        if max_correct:
            for i in range(reps):
                # permute indexing vector
                prng.shuffle(rr)
                # grab shuffled values
                pp = np.take(potential_outcomes_all, rr, axis=0)
                # calculate statistic
                curr_tst = tst_stat(pp[:nx, :, 0], pp[nx:, :, 1])
                # grab the most extreme statistic observed across all tests
                curr_max = max(curr_tst.min(), curr_tst.max(), key=abs)
                # count if observed statistic is larger or smaller than permuted
                hitsUp += curr_max >= tst
                hitsDn += curr_max <= tst
            # calculate percentile
            pUp = hitsUp/(reps+plus1)
            pDn = hitsDn/(reps+plus1)
            return thePvalue[alternative](pUp, pDn)
        else:
            for i in range(reps):
                # permute indexing vector
                prng.shuffle(rr)
                # grab shuffled values
                pp = np.take(potential_outcomes_all, rr, axis=0)
                # count if observed statistic is larger or smaller than permuted
                hitsUp += tst_stat(pp[:nx, :, 0], pp[nx:, :, 1]) >= tst
                hitsDn += tst_stat(pp[:nx, :, 0], pp[nx:, :, 1]) <= tst
            # calculate percentile
            pUp = hitsUp/(reps+plus1)
            pDn = hitsDn/(reps+plus1)
        return thePvalue[alternative](pUp, pDn)



def multitest_one_sample(x, y=None, reps=10**5, stat='mean', alternative="greater",
               keep_dist=False, seed=None, plus1=True,max_correct=False):
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
        Sample 1 with shape (observations,tests)
    y : array-like
        Sample 2 with shape (observations,tests). Must preserve the order of pairs with x.
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
    # since one sample test, if we get both x and y turn into one sample
    # (and ensure they have the same shape)
    if y is None:
        z = x
    # make sure
    elif x.shape != y.shape:
        raise ValueError('x and y must have the same shape')
    else:
        z = np.array(x) - np.array(y)
    # get number of hypotheses being tested
    num_tests = z.shape[1]
    # create dictionary to store functions for calculating p values
    thePvalue = {
        'greater': lambda pUp, pDn: pUp+plus1/(reps+plus1),
        'less': lambda pUp, pDn: pDn+plus1/(reps+plus1),
        'two-sided': lambda pUp, pDn: 2 * np.min(np.stack([0.5*np.ones(num_tests), \
                                    pUp+plus1/(reps+plus1), \
                                    pDn+plus1/(reps+plus1)],1),axis=1)
    }
    # create dictionary to store common statistics ensuring correct axis
    stats = {
        'mean': lambda u: np.mean(u,axis=0),
        't': lambda u: ttest_1samp(u, 0, axis=0)[0]
    }
    # if we were given a custom statistic function, use that
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]
    # calculate observed statistic
    tst = tst_fun(z)
    # account for all combinations of keep_dist (keep distribution)
    # and max_correct (create max distribution to correct for multiple 
    # hypothesis testing) 
    if keep_dist:
        if max_correct:
            # preallocate space to build null distribution 
            # (1D since only taking extreme values)
            dist = np.empty(reps)
            for i in range(reps):
                # calculate statistic of current permutation
                curr_tst = tst_fun(z * (1 - 2 * prng.randint(0, 2, z.shape)))
                # grab the most extreme value across tests
                dist[i] = max(curr_tst.min(), curr_tst.max(), key=abs)
            # calculate percentile for each test
            pUp = np.empty(num_tests)
            pDn = np.empty(num_tests)
            for i in range(num_tests):
                pUp[i] = np.sum(dist >= tst[i])/(reps+plus1)
                pDn[i] = np.sum(dist <= tst[i])/(reps+plus1)
            return thePvalue[alternative](pUp, pDn), tst, dist
        else:
            # preallocate space to build null distribution 
            # (2D since each test will have its own distribution)
            dist = np.empty((reps,num_tests))
            for i in range(reps):
                 # calculate statistic of current permutation
                dist[i,:] = tst_fun(z * (1 - 2 * prng.randint(0, 2, z.shape)))
            # calculate percentile for each test
            pUp = np.sum(dist >= tst,axis=0)/(reps+plus1)
            pDn = np.sum(dist <= tst,axis=0)/(reps+plus1)
            return thePvalue[alternative](pUp, pDn), tst, dist
    else:
        # preallocate vectors to store number of times our observed statistic
        # is greater than the permuted statistic(s). Keeps memory requirement low.
        hitsUp = np.zeros(num_tests)
        hitsDn = np.zeros(num_tests)
        if max_correct:
            for i in range(reps):
                # calculate statistic for current permutation
                curr_tst = tst_fun(z * (1 - 2 * prng.randint(0, 2, z.shape)))
                # grab the most extreme statistic across all tests
                curr_max = max(curr_tst.min(), curr_tst.max(), key=abs)
                # iterate counters accordingly
                hitsUp += curr_max >= tst
                hitsDn += curr_max <= tst
            # calculate percentiles
            pUp = hitsUp/(reps+plus1)
            pDn = hitsDn/(reps+plus1)
            return thePvalue[alternative](pUp, pDn), tst
        else:
            for i in range(reps):
                # calculate statistic for current permutation
                curr_tst = tst_fun(z * (1 - 2 * prng.randint(0, 2, z.shape)))
                # iterate counters accordingly
                hitsUp += curr_tst >= tst
                hitsDn += curr_tst <= tst
            # calculate percentiles
            pUp = hitsUp/(reps+plus1)
            pDn = hitsDn/(reps+plus1)
        return thePvalue[alternative](pUp, pDn), tst


def multitest_two_sample(x, y, reps=10**5, stat='mean', alternative="greater",
               keep_dist=False, seed=None, plus1=True, max_correct=False):
    r"""
    One-sided or two-sided, two-sample permutation multi-test for equality of
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
        Sample 1 with shape (observations,tests)
    y : array-like
        Sample 2 with shape (observations,tests)
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
    # ensure x and y have the same shape
    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape')
    # Set up potential outcomes; under the null, all units are exchangeable
    pot_out_all = np.stack(
        [np.concatenate([x, y]), np.concatenate([x, y])],2)
    
    
    # If stat is callable, use it as the test function. Otherwise, look in the
    # dictionary
    stats = {
        'mean': lambda u, v: np.mean(u,axis=0) - np.mean(v,axis=0),
        't': lambda u, v: ttest_ind(u, v, axis=0, equal_var=True)[0]
    }
    
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]
    
    # get number of observations
    nx = x.shape[0]
    # calculate observed statistic for all tests
    observed_tst = tst_fun(pot_out_all[:nx, :, 0], pot_out_all[nx:, :, 1])
    # call main worker function
    res = multitest_two_sample_core(pot_out_all, nx, tst_fun, alternative=alternative,
                          reps=reps, keep_dist=keep_dist, seed=seed, plus1=plus1, max_correct=max_correct)
    # accomadate user request for returning distribution 
    if keep_dist:
        return res[0], observed_tst, res[1]
    else:
        return res, observed_tst


def multitest_potential_outcomes(x, y, f, finverse):
    """
    Given observations $x$ under treatment and $y$ under control conditions,
    returns the potential outcomes for units under their unobserved condition
    under the hypothesis that $x_i = f(y_i)$ for all units.

    Parameters
    ----------
    x : array-like
        Outcomes under treatment
    y : array-like
        Outcomes under control
    f : function
        An invertible function
    finverse : function
        The inverse function to f.

    Returns
    -------
    potential_outcomes : 2D array
        The first column contains all potential outcomes under the treatment,
        the second column contains all potential outcomes under the control.
    """
    
    tester = np.array(range(5)) + 1
    assert np.allclose(finverse(f(tester)),
                       tester), "f and finverse aren't inverses"
    assert np.allclose(f(finverse(tester)),
                       tester), "f and finverse aren't inverses"
    
    pot_treat = np.concatenate([x, f(y)])
    pot_ctrl = np.concatenate([finverse(x), y])
    
    return np.stack([pot_treat, pot_ctrl],2)


# def multitest_two_sample_shift(x, y, reps=10**5, stat='mean', alternative="greater",
#                      keep_dist=False, seed=None, shift=None, plus1=True):
#     r"""
#     One-sided or two-sided, two-sample permutation multi-test for equality of
#     two means, with p-value estimated by simulated random sampling with
#     reps replications.

#     Tests the hypothesis that x and y are a random partition of x,y
#     against the alternative that x comes from a population with mean

#     (a) greater than that of the population from which y comes,
#         if side = 'greater'
#     (b) less than that of the population from which y comes,
#         if side = 'less'
#     (c) different from that of the population from which y comes,
#         if side = 'two-sided'

#     If ``keep_dist``, return the distribution of values of the test statistic;
#     otherwise, return only the number of permutations for which the value of
#     the test statistic and p-value.

#     Parameters
#     ----------
#     x : array-like
#         Sample 1
#     y : array-like
#         Sample 2
#     reps : int
#         number of repetitions
#     stat : {'mean', 't'}
#         The test statistic.

#         (a) If stat == 'mean', the test statistic is (mean(x) - mean(y))
#             (equivalently, sum(x), since those are monotonically related)
#         (b) If stat == 't', the test statistic is the two-sample t-statistic--
#             but the p-value is still estimated by the randomization,
#             approximating the permutation distribution.
#             The t-statistic is computed using scipy.stats.ttest_ind
#         (c) If stat is a function (a callable object), the test statistic is
#             that function. The function should take two arguments:
#             given a permutation of the pooled data, the first argument is the
#             "new" x and the second argument is the "new" y.
#             For instance, if the test statistic is the Kolmogorov-Smirnov distance 
#             between the empirical distributions of the two samples, 
#             $\max_t |F_x(t) - F_y(t)|$, the test statistic could be written:

#             f = lambda u, v: np.max( \
#                 [abs(sum(u<=val)/len(u)-sum(v<=val)/len(v)) for val in np.concatenate([u, v])]\
#                 )
                
#     alternative : {'greater', 'less', 'two-sided'}
#         The alternative hypothesis to test
#     keep_dist : bool
#         flag for whether to store and return the array of values
#         of the irr test statistic
#     seed : RandomState instance or {None, int, RandomState instance}
#         If None, the pseudorandom number generator is the RandomState
#         instance used by `np.random`;
#         If int, seed is the seed used by the random number generator;
#         If RandomState instance, seed is the pseudorandom number generator
#     shift : float
#         The relationship between x and y under the null hypothesis.

#         (a) A constant scalar shift in the distribution of y. That is, x is equal
#             in distribution to y + shift.
#         (b) A tuple containing the function and its inverse $(f, f^{-1})$, so
#             $x_i = f(y_i)$ and $y_i = f^{-1}(x_i)$
#     plus1 : bool
#         flag for whether to add 1 to the numerator and denominator of the
#         p-value based on the empirical permutation distribution. 
#         Default is True.

#     Returns
#     -------
#     float
#         the estimated p-value
#     float
#         the test statistic
#     list
#         The distribution of test statistics.
#         These values are only returned if `keep_dist` == True
#     """
#     # Set up potential outcomes according to shift
#     if isinstance(shift, float) or isinstance(shift, int):
#         # Potential outcomes for all units under treatment
#         pot_outx = np.concatenate([x, y + shift])
#         # Potential outcomes for all units under control
#         pot_outy = np.concatenate([x - shift, y])
#         pot_out_all = np.stack([pot_outx, pot_outy],2)
#     elif isinstance(shift, tuple):
#         assert (callable(shift[0])), "Supply f and finverse in shift tuple"
#         assert (callable(shift[1])), "Supply f and finverse in shift tuple"
#         pot_out_all = multitest_potential_outcomes(x, y, shift[0], shift[1])
#     else:
#         raise ValueError("Bad input for shift")
#     # If stat is callable, use it as the test function. Otherwise, look in the
#     # dictionary
#     stats = {
#         'mean': lambda u, v: np.mean(u,axis=0) - np.mean(v,axis=0),
#         't': lambda u, v: ttest_ind(u, v, axis=0, equal_var=True)[0]
#     }
#     if callable(stat):
#         tst_fun = stat
#     else:
#         tst_fun = stats[stat]
#     # get number of observations
#     nx = x.shape[0]
#     # calculate observed statistics for all tests
#     observed_tst = tst_fun(pot_out_all[:nx,:, 0], pot_out_all[nx:,:, 1])
#     # call main worker function
#     res = multitest_two_sample_core(pot_out_all, nx, tst_fun, alternative=alternative,
#                           reps=reps, keep_dist=keep_dist, seed=seed, plus1=plus1)
#     # accomadate user request for returning distribution 
#     if keep_dist:
#         return res[0], observed_tst, res[1]
#     else:
#         return res, observed_tst





