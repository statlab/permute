# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:55:37 2022

@author: Clayton
"""

import numpy as np
from numpy.random import RandomState

import pytest

from ..stratified import multitest_stratified_permutationtest as spt
from ..stratified import multitest_stratified_permutationtest_mean as sptm
from ..stratified import multitest_stratified_corrcoef, multitest_stratified_sim_corr, multitest_stratified_two_sample

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
