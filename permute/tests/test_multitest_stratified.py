# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:55:37 2022

@author: Clayton
"""

import numpy as np
from numpy.random import RandomState

import pytest

from ..multitest_stratified import multitest_stratified_permutationtest as spt
from ..multitest_stratified import multitest_stratified_permutationtest_mean as sptm
from ..multitest_stratified import (multitest_stratified_corrcoef, multitest_stratified_sim_corr,
                                    multitest_stratified_two_sample)

def test_multitest_stratified_permutationtest():
    num_tests = 2
    group = np.repeat([1, 2, 3], 9)
    condition = np.repeat([1, 2, 3] * 3, 3)
    response = np.zeros((group.shape[0],num_tests))
    response[[0, 1, 3, 9, 10, 11, 18, 19, 20],:] = 1
    
    res = spt(group, condition, response, reps=1000, seed=42)
    res1 = spt(group, condition, response, alternative='less', reps=1000, seed=42)
    assert np.all(res[0] < 0.01)
    assert np.all(res[1] == res1[1])
    np.testing.assert_almost_equal(res[0], 1-res1[0])
    res2 = spt(group, condition, response, alternative='two-sided', reps=1000, seed=42)
    assert np.all(res2[0] < 0.02)
    
    group = np.array([1, 1, 1])
    condition = np.array([2, 2, 2])
    response = np.zeros((group.shape[0],num_tests))
    res2 = spt(group, condition, response, reps=1000, seed=42)
    assert res2 == (1.0, np.nan, None)


def test_multitest_stratified_permutationtest_mean():
    num_tests = 2
    group = np.array([1, 2, 1, 2])
    condition = np.array([1, 1, 2, 2])
    response = np.zeros((group.shape[0],num_tests))
    groups = np.unique(group)
    conditions = np.unique(condition)
    res = sptm(group, condition, response, groups, conditions)
    assert np.all(res == (0.0,0.0))
    res2 = sptm(group, condition, response)  # check defaults work
    assert np.all(res2 == (0.0,0.0))


def test_multitest_stratified_permutationtest_mean_error():
    num_tests = 2
    group = np.array([1, 1, 1])
    condition = np.array([2, 2, 2])
    response = np.zeros((group.shape[0],num_tests))
    groups = np.unique(group)
    conditions = np.unique(condition)
    pytest.raises(ValueError, sptm, group, condition, response, groups, conditions)


def test_multitest_stratified_corrcoef():
    num_tests = 2
    prng = RandomState(42)
    x = prng.rand(10,num_tests)
    y = x
    group = prng.randint(3, size=10)
    res1 = multitest_stratified_corrcoef(x, y, group)
    res2 = multitest_stratified_corrcoef(x, y, group)
    assert np.all(res1 == res2)


def test_multitest_stratified_sim_corr():
    num_tests = 2
    prng = RandomState(42)
    x = prng.rand(10,num_tests)
    y = x
    group = prng.randint(0, 3, size=10)
    res1 = multitest_stratified_sim_corr(x, y, group, seed=prng, reps=100)
    res2 = multitest_stratified_sim_corr(x, y, group, seed=prng, alternative='less', reps=100)
    res3 = multitest_stratified_sim_corr(x, y, group, seed=prng, alternative='two-sided', reps=100)
    
    np.testing.assert_almost_equal(res1[0], 1-res2[0])
    assert np.all(res1[1] == res2[1])
    assert np.all(res1[1] == res3[1])
    assert np.all(2*res1[0] == res3[0])


def test_multitest_stratified_strat_tests_equal():
    num_tests = 2
    group = np.repeat([1, 2, 3], 10)
    condition = np.repeat([1, 2] * 3, 5)
    response = np.zeros((group.shape[0],num_tests))
    response[[0, 1, 3, 9, 10, 11, 18, 19, 20],:] = 1
    
    res1 = spt(group, condition, response, reps=1000, seed=42)
    res2 = multitest_stratified_two_sample(group, condition, response, reps=1000,
                                stat='mean_within_strata', seed=42)
    assert np.all(res1[1] == res2[1])
    assert np.all(np.fabs(res1[0]-res2[0]) < 0.05)
    
def test_multitest_stratified_two_sample():
    num_tests = 2
    group = np.repeat([1, 2, 3], 10)
    condition = np.repeat([1, 2] * 3, 5)
    response = np.zeros((group.shape[0],num_tests))
    response[[0, 1, 3, 9, 10, 11, 18, 19, 20],:] = 1
    
    res = multitest_stratified_two_sample(group, condition, response, reps=1000,
                                stat='mean', seed=42)
    np.testing.assert_almost_equal(res[0], 0.245, 2)
    assert np.all(res[1] == 0.2)
    
    (p, t, dist) = multitest_stratified_two_sample(group, condition, response, reps=1000,
                                stat='mean', seed=42, keep_dist=True)
    assert np.all(res[0] == p)
    assert np.all(res[1] == t)
    
    stat_fun = lambda u: sptm(group, condition, u, np.unique(group), np.unique(condition))
    res = multitest_stratified_two_sample(group, condition, response, reps=1000,
                                stat=stat_fun, seed=42)
    # below differs from test_stratified because changed dependence of stat to use
    # in multitest_stratified_two_sample from number of unique groups to conditions. 
    # TODO ensure this is appropriate
    np.testing.assert_almost_equal(res[0], 0.6733, 3)
    np.testing.assert_almost_equal(res[1], -0.2, 3)


def _null_design(seed=0, num_tests=3):
    """Four complete strata (2 + 2 observations per condition) with no effect."""
    rng = np.random.default_rng(seed)
    group = np.repeat([1, 2, 3, 4], 4)
    condition = np.tile([1, 1, 2, 2], 4)
    response = rng.normal(size=(group.shape[0], num_tests))
    return group, condition, response


def test_multitest_stratified_incomplete_stratum_is_skipped():
    # A stratum observed in only one condition is uninformative under
    # within-stratum permutation. It used to make every statistic NaN, and
    # since NaN >= x is False, every p-value came back at its minimum.
    group, condition, response = _null_design()
    p_ref, tst_ref, _ = spt(group, condition, response, alternative='two-sided', reps=500, seed=1)
    group2 = np.append(group, [9, 9])
    condition2 = np.append(condition, [1, 1])
    response2 = np.vstack([response, [[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]])
    p, tst, _ = spt(group2, condition2, response2, alternative='two-sided', reps=500, seed=1)
    assert np.all(np.isfinite(tst))
    np.testing.assert_allclose(tst, tst_ref)
    assert np.all(p > 1 / 501 + 1e-12)


def test_multitest_stratified_mean_skips_incomplete_strata():
    group, condition, response = _null_design()
    full = sptm(group, condition, response)
    group2 = np.append(group, [9])
    condition2 = np.append(condition, [2])
    response2 = np.vstack([response, [[1.0, 1.0, 1.0]]])
    np.testing.assert_allclose(sptm(group2, condition2, response2), full)


def test_multitest_stratified_permutationtest_no_complete_strata():
    group = np.array([1, 1, 2, 2])
    condition = np.array([1, 1, 2, 2])
    response = np.zeros((4, 2))
    pytest.raises(ValueError, spt, group, condition, response, reps=10)


def test_multitest_stratified_nan_response_gives_nan_pvalue():
    group, condition, response = _null_design()
    response[0, 1] = np.nan
    p, tst, _ = spt(group, condition, response, alternative='two-sided', reps=500, seed=1)
    assert np.isnan(p[1]) and np.isnan(tst[1])
    assert np.all(np.isfinite(p[[0, 2]]))
    for alt in ('greater', 'less'):
        assert np.isnan(spt(group, condition, response, alternative=alt, reps=200, seed=1)[0][1])


def test_multitest_stratified_sim_corr_nan_gives_nan_pvalue():
    rng = np.random.default_rng(2)
    group = np.repeat([1, 2], 10)
    x = rng.normal(size=(20, 2))
    y = rng.normal(size=(20, 2))
    x[3, 0] = np.nan
    p, tst, _ = multitest_stratified_sim_corr(x, y, group, reps=200, seed=3)
    assert np.isnan(p[0]) and np.isfinite(p[1])


def test_multitest_stratified_two_sample_nan_gives_nan_pvalue():
    group, condition, response = _null_design()
    response[2, 0] = np.nan
    for keep_dist in (True, False):
        res = multitest_stratified_two_sample(group, condition, response, stat='mean_within_strata',
                                              reps=200, keep_dist=keep_dist, seed=4)
        assert np.isnan(res[0][0]) and np.all(np.isfinite(res[0][1:]))
