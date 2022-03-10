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
    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape')
    num_tests = x.shape[1]
    corr_mat_mask = np.zeros((2*num_tests,2*num_tests),dtype=bool)
    corr_mat_mask[x.shape[1]+np.arange(num_tests),np.arange(num_tests)] = True
    tst = np.zeros(num_tests)
    for g in np.unique(group):
        gg = group == g
        tst += np.corrcoef(x[gg,:], y[gg,:],rowvar=False)[corr_mat_mask]
    return tst



def multitest_stratified_sim_corr(x, y, group, reps=10**4, alternative='greater', seed=None, plus1=True, max_correct=False):
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
    max_correct : bool
        flag for whether to perform max statistic multiple testing
        correction. Builds the null distribution from the most extreme value
        across tests for each iteration of the permutation. Default is False.
    
    Returns
    -------
    float
      the estimated p-value
    float
      the observed test statistic
    list
      the null distribution
    """
    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape')
    num_tests = x.shape[1]
    prng = get_prng(seed)
    x = x.astype(float)
    y = y.astype(float)
    tst = multitest_stratified_corrcoef(x, y, group)
    if max_correct:
        dist = np.empty(reps)
        for i in range(reps):
            curr_tst = multitest_stratified_corrcoef(permute_within_groups(x, group, prng), y, group)
            dist[i] = max(curr_tst.min(), curr_tst.max(), key=abs)
        right_pv = np.empty(num_tests)
        for i in range(num_tests):
            right_pv[i] = np.sum(dist >= tst[i]) / (reps+plus1)
    else:
        dist = [multitest_stratified_corrcoef(permute_within_groups(x, group, prng), y, group)
            for i in range(reps)]
        right_pv = np.sum(dist >= tst,axis=0) / (reps+plus1)
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
    num_tests = response.shape[1]
    if groups is None:
        groups = np.unique(group)
    if conditions is None:
        conditions = np.unique(condition)
    tst = np.zeros(num_tests)
    if len(groups) < 2:
        raise ValueError('Number of groups must be at least 2.')
    elif len(groups) == 2:
        stat = lambda u: np.fabs(u[0] - u[1])
        for g in groups:
            gg = group == g
            x = [gg & (condition == c) for c in conditions]
            tst += stat([response[x[j],:].mean(axis=0) for j in range(len(x))])
    elif len(groups) > 2:
        for g in groups:
            gg = group == g
            x = [gg & (condition == c) for c in conditions]
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
        plus1=True,
        max_correct=False):
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
    max_correct : bool
        flag for whether to perform max statistic multiple testing
        correction. Builds the null distribution from the most extreme value
        across tests for each iteration of the permutation. Default is False.
    
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
    num_tests = response.shape[1]
    groups = np.unique(group)
    conditions = np.unique(condition)
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
    thePvalue = {
        'greater': lambda p: p + plus1/(reps+plus1),
        'less': lambda p: 1 - (p + plus1/(reps+plus1)),
        'two-sided': lambda p: 2 * np.min([p + plus1/(reps+plus1), 
                                           1 - (p + plus1/(reps+plus1))],axis=0)
    }
    
    if len(conditions) < 2:
        return 1.0, np.nan, None
    else:
        tst = tst_fun(condition)
        if max_correct:
            dist = np.zeros(reps)
            for i in range(int(reps)):
                curr_tst = tst_fun(permute_within_groups(condition, group, prng))
                dist[i] = max(curr_tst.min(), curr_tst.max(), key=abs)
            right_pv = np.empty(num_tests)
            for i in range(num_tests):
                right_pv[i] = np.sum(dist >= tst[i])/(reps+plus1)
            return thePvalue[alternative](right_pv), tst, dist
        else:
            dist = np.zeros((reps,num_tests))
            for i in range(int(reps)):
                dist[i,:] = tst_fun(permute_within_groups(condition, group, prng))
            
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
        plus1=True,
        max_correct=False):
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
    max_correct : bool
        flag for whether to perform max statistic multiple testing
        correction. Builds the null distribution from the most extreme value
        across tests for each iteration of the permutation. Default is False.

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
    num_tests = response.shape[1]
    ordering = condition.argsort()
    response = response[ordering]
    condition = condition[ordering]
    group = group[ordering]
    ntreat = np.sum(condition == condition[0])
    
    groups = np.unique(group)
    conditions = np.unique(condition)
    # If stat is callable, use it as the test function. Otherwise, look in the
    # dictionary
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
        
    thePvalue = {
        'greater': lambda p: p + plus1/(reps+plus1),
        'less': lambda p: 1 - (p + plus1/(reps+plus1)),
        'two-sided': lambda p: 2 * np.min([p + plus1/(reps+plus1), 
                                           1 - (p + plus1/(reps+plus1))],axis=0)
    }
    observed_tst = tst_fun(response)
    
    if keep_dist:
        if max_correct:
            dist = np.empty(reps)
            for i in range(reps):
                curr_tst = tst_fun(permute_within_groups(
                    response, group, seed=prng))
                dist[i] = max(curr_tst.min(), curr_tst.max(), key=abs)
            hits = np.empty(num_tests)
            for i in range(num_tests):
                hits[i] = np.sum(dist >= observed_tst[i])
                return thePvalue[alternative](hits / (reps+plus1)), observed_tst, dist
        else:
            dist = np.empty((reps,num_tests))
            for i in range(reps):
                dist[i,:] = tst_fun(permute_within_groups(
                    response, group, seed=prng))
            hits = np.sum(dist >= observed_tst,axis=0)
            return thePvalue[alternative](hits / (reps+plus1)), observed_tst, dist
    else:
        if max_correct:
            hits = np.sum([(tst_fun(permute_within_groups(
            response, group, seed=prng)) >= observed_tst)
                 for i in range(reps)],axis=0)
            return thePvalue[alternative](hits / (reps+plus1)), observed_tst
        else:
            hits = np.zeros(num_tests)
            for i in range(reps):
                curr_tst = tst_fun(permute_within_groups(response, group, seed=prng))
                hits += curr_tst >= observed_tst
            return thePvalue[alternative](hits / (reps+plus1)), observed_tst
