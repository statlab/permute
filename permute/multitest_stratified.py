# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:31:17 2022

@author: Clayton
"""

import numpy as np
from scipy.stats import ttest_ind

from .utils import get_prng, permute_within_groups

def multitest_stratified_corrcoef(x, y, group):
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
    # ensure x and y are the same shape (same number of observations and tests)
    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape')
    # get number of hypotheses tested
    num_tests = x.shape[1]
    # create mask to grab correlations from corrcoeff we care about (don't care about all pairs)
    corr_mat_mask = np.zeros((2*num_tests,2*num_tests),dtype=bool)
    corr_mat_mask[x.shape[1]+np.arange(num_tests),np.arange(num_tests)] = True
    # preallocate vector to store aggregate correlations for each test
    tst = np.zeros(num_tests)
    for g in np.unique(group):
        # create mask for current group
        gg = group == g
        # calculate and grab correlation coefficients for current group
        tst += np.corrcoef(x[gg,:], y[gg,:],rowvar=False)[corr_mat_mask]
    return tst



def multitest_stratified_sim_corr(x, y, group, reps=10**4, alternative='greater', seed=None, plus1=True):
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
    plus1 : bool
        flag for whether to add 1 to the numerator and denominator of the
        p-value based on the empirical permutation distribution. 
        Default is True.
    
    Returns
    -------
    float
      the estimated p-value
    float
      the observed test statistic
    list
      the null distribution
    """
    # ensure x and y have the same shape (same number of observations and tests)
    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape')
    prng = get_prng(seed)
    x = x.astype(float)
    y = y.astype(float)
    # calculate observed statistic
    tst = multitest_stratified_corrcoef(x, y, group)
    # account for user wanting to perform max correction
    # calculate statistic on each permutation to build null distribution
    dist = [multitest_stratified_corrcoef(permute_within_groups(x, group, prng), y, group)
        for i in range(reps)]
    # calculate percentile for each test
    right_pv = np.sum(dist >= tst,axis=0) / (reps+plus1)
    # create dictionary to store p value calculations
    thePvalue = {
        'greater': lambda p: p + plus1/(reps+plus1),
        'less': lambda p: 1 - (p + plus1/(reps+plus1)),
        'two-sided': lambda p: 2 * np.min([p + plus1/(reps+plus1), 
                                           1 - (p + plus1/(reps+plus1))],axis=0)
    }
    return thePvalue[alternative](right_pv), tst, dist


def multitest_stratified_permutationtest_mean(group, condition, response,
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
        Responses, of the same length as group. Has shape [observations,tests]..
    groups : array-like
        Group labels. By default, it is the unique values of group
    conditions : array-like
        Condition labels. By default, it is the unique values of condition


    Returns
    -------
    tst : float
      The observed test statistic
    """
    # get number of hypotheses to test
    num_tests = response.shape[1]
    # get the ID for each group
    if groups is None:
        groups = np.unique(group)
    # get the ID for each condition
    if conditions is None:
        conditions = np.unique(condition)
    # preallocate vector to store the aggregate statistic for each test
    tst = np.zeros(num_tests)
    # check there are at least 2 groups
    if len(groups) < 2:
        raise ValueError('Number of groups must be at least 2.')
    # if 2 conditions, calculate mean. If more than 2, calculate std of outcomes
    # TODO ensure this is intended behavior, in stratified.py this is done 
    # with the variable "groups", but that doesn't seem right to me
    elif len(conditions) == 2:
        stat = lambda u: u[0] - u[1]
        for g in groups:
            # create mask for current group
            gg = group == g
            # create conjugate mask for group and condition
            x = [gg & (condition == c) for c in conditions]
            # aggregate statistic for each group and condition
            tst += stat([response[x[j],:].mean(axis=0) for j in range(len(x))])
    elif len(conditions) > 2:
        for g in groups:
            # create mask for current group
            gg = group == g
            # create conjugate mask for group and condition
            x = [gg & (condition == c) for c in conditions]
            # aggregate statistic for each group and condition
            tst += np.std([response[x[j],:].mean(axis=0) for j in range(len(x))],0)
    return tst


def multitest_stratified_permutationtest(
        group,
        condition,
        response,
        alternative='greater',
        reps=10**5,
        testStatistic='mean',
        seed=None,
        plus1=True):
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
        Function to compute test statistic. By default,
        stratified_permutationtest_mean
        The test statistic. Either a string or function.

        (a) If stat == 'mean', the test statistic is
            stratified_permutationtest_mean (default).
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
    plus1 : bool
        flag for whether to add 1 to the numerator and denominator of the
        p-value based on the empirical permutation distribution. 
        Default is True.
    
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
    # get the number of hypotheses to test
    num_tests = response.shape[1]
    # get the group IDs
    groups = np.unique(group)
    # get the condition IDs
    conditions = np.unique(condition)
    # create a dictionary to store common statistic calculation
    stats = {
        'mean': lambda u: multitest_stratified_permutationtest_mean(
            group,
            u,
            response,
            groups,
            conditions)}
    if callable(testStatistic):
        tst_fun = testStatistic
    else:
        tst_fun = stats[testStatistic]
        # create dictionary to store p values calculatoins
    thePvalue = {
        'greater': lambda p: p + plus1/(reps+plus1),
        'less': lambda p: 1 - (p + plus1/(reps+plus1)),
        'two-sided': lambda p: 2 * np.min([p + plus1/(reps+plus1), 
                                           1 - (p + plus1/(reps+plus1))],axis=0)
    }
    #
    if len(conditions) < 2:
        # TODO would it be more appropriate to raise error?
        # raise ValueError('Number of conditions must be at least 2.')
        return 1.0, np.nan, None
    else:
        # calculate observed statistic
        tst = tst_fun(condition)
        # preallocate vector to store null distribution
        # (2D because each test will have its own distribution)
        dist = np.zeros((reps,num_tests))
        for i in range(int(reps)):
            # calculate statistic for current permutation
            dist[i,:] = tst_fun(permute_within_groups(condition, group, prng))
        # calculate percentile for each test
        right_pv = np.sum(dist >= tst,axis=0) / (reps+plus1)
        return thePvalue[alternative](right_pv), tst, dist


def multitest_stratified_two_sample(
        group,
        condition,
        response,
        stat='mean',
        alternative="greater",
        reps=10**5,
        keep_dist=False,
        seed=None,
        plus1=True):
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

    Permutations are carried out within the given groups.  Under the null
    hypothesis, observations within each group are exchangeable.

    If ``keep_dist``, return the distribution of values of the test statistic;
    otherwise, return only the number of permutations for which the value of
    the test statistic and p-value.

    Parameters
    ----------
    group : array-like
        Group memberships
    condition : array-like
        Treatment conditions, of the same length as group
    response : array-like
        Responses, of the same length as group
    stat : {'mean', 't'}
        The test statistic.

        (a) If stat == 'mean', the test statistic is (mean(x) - mean(y))
            (equivalently, sum(x), since those are monotonically related),
            omitting NaNs, which therefore can be used to code non-responders
        (b) If stat == 't', the test statistic is the two-sample t-statistic--
            but the p-value is still estimated by the randomization,
            approximating the permutation distribution.
            The t-statistic is computed using scipy.stats.ttest_ind
        (c) If stat == 'mean_within_strata', the test statistic is the
            difference in means within each stratum, added across strata.
        (d) If stat is a function (a callable object), the test statistic is
            that function.  The function should take a permutation of the
            pooled data and compute the test function from it. For instance,
            if the test statistic is the Kolmogorov-Smirnov distance between
            the empirical distributions of the two samples,
            $max_t |F_x(t) - F_y(t)|$, the test statistic could be written:

            f = lambda u: np.max( \
                [abs(sum(u[:len(x)]<=v)/len(x)-sum(u[len(x):]<=v)/len(y))
                for v in u]\
                )
    alternative : {'greater', 'less', 'two-sided'}
        The alternative hypothesis to test
    reps : int
        Number of permutations
    keep_dist : bool
        flag for whether to store and return the array of values
        of the test statistic
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator.
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
    # get number of hypotheses to test
    num_tests = response.shape[1]
    # get indexing to sort by condition (not sure why this is necessary)
    ordering = condition.argsort()
    response = response[ordering]
    condition = condition[ordering]
    group = group[ordering]
    # get number of samples that received condition with lowest ID
    # TODO should we ensure each condition has the same number of samples?
    ntreat = np.sum(condition == condition[0])
    
    # get the IDs for each group and condition
    groups = np.unique(group)
    conditions = np.unique(condition)
    # If stat is callable, use it as the test function. Otherwise, look in the
    # dictionary
    # TODO there is no x, not sure what desired behavior is here
    stats = {
        'mean': lambda u: np.nanmean(u[:ntreat],axis=0) - np.nanmean(u[ntreat:],axis=0),
        't': lambda u: ttest_ind(
            u[:len(x)][~np.isnan(u[:ntreat])],
            u[len(x):][~np.isnan(u[ntreat:])],
            axis=0,equal_var=True)[0],
        'mean_within_strata': lambda u: multitest_stratified_permutationtest_mean(group,
                                                                        condition,
                                                                        u,
                                                                        groups,
                                                                        conditions)
    }
    if callable(stat):
        tst_fun = stat
    else:
        tst_fun = stats[stat]
    # create dictionary to store p value calculations
    thePvalue = {
        'greater': lambda p: p + plus1/(reps+plus1),
        'less': lambda p: 1 - (p + plus1/(reps+plus1)),
        'two-sided': lambda p: 2 * np.min([p + plus1/(reps+plus1), 
                                           1 - (p + plus1/(reps+plus1))],axis=0)
    }
    # get observed statistic
    observed_tst = tst_fun(response)
    # account for keep_dist (keep distribution) 
    if keep_dist:
        # preallocate vector for null distribution
        # (2D because build null distribution for each test)
        dist = np.empty((reps,num_tests))
        for i in range(reps):
            # calculate statistic for current permutation
            dist[i,:] = tst_fun(permute_within_groups(
                response, group, seed=prng))
        # calculate percentile for each test
        hits = np.sum(dist >= observed_tst,axis=0)
        return thePvalue[alternative](hits / (reps+plus1)), observed_tst, dist
    else:
        # create vector to store number of times each hypothesis is less
        # than the corresponding statistic of the permuted values
        hits = np.zeros(num_tests)
        for i in range(reps):
            # calculate current statistic
            curr_tst = tst_fun(permute_within_groups(response, group, seed=prng))
            hits += curr_tst >= observed_tst
        return thePvalue[alternative](hits / (reps+plus1)), observed_tst
