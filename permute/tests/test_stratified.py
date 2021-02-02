import numpy as np
import math
from numpy.random import RandomState

import pytest
from cryptorandom.cryptorandom import SHA256

from ..stratified import stratified_permutationtest as spt
from ..stratified import stratified_permutationtest_mean as sptm
from ..stratified import corrcoef, sim_corr, stratified_two_sample


def test_stratified_permutationtest():
    group = np.repeat([1, 2, 3], 9)
    condition = np.repeat([1, 2, 3] * 3, 3)
    response = np.zeros_like(group)
    response[[0, 1, 3, 9, 10, 11, 18, 19, 20]] = 1

    res = spt(group, condition, response, reps=1000, seed=42)
    res1 = spt(group, condition, response, alternative='less', reps=1000, seed=42)
    assert res[0] < 0.01
    assert res[1] == res1[1]
    np.testing.assert_almost_equal(res[0], 1-res1[0])
    res2 = spt(group, condition, response, alternative='two-sided', reps=1000, seed=42)
    assert res2[0] < 0.02

    group = np.array([1, 1, 1])
    condition = np.array([2, 2, 2])
    response = np.zeros_like(group)
    res2 = spt(group, condition, response, reps=1000, seed=42)
    assert res2 == (1.0, np.nan, None)


def test_stratified_permutationtest_mean():
    group = np.array([1, 2, 1, 2])
    condition = np.array([1, 1, 2, 2])
    response = np.zeros_like(group)
    groups = np.unique(group)
    conditions = np.unique(condition)
    res = sptm(group, condition, response, groups, conditions)
    assert res == 0.0
    res2 = sptm(group, condition, response)  # check defaults work
    assert res2 == 0.0


def test_stratified_permutationtest_mean_error():
    group = np.array([1, 1, 1])
    condition = np.array([2, 2, 2])
    response = np.zeros_like(group)
    groups = np.unique(group)
    conditions = np.unique(condition)
    pytest.raises(ValueError, sptm, group, condition, response, groups, conditions)


def test_corrcoef():
    prng = RandomState(42)
    x = prng.rand(10)
    y = x
    group = prng.randint(3, size=10)
    res1 = corrcoef(x, y, group)
    res2 = corrcoef(x, y, group)
    assert res1 == res2


def test_sim_corr():
    prng = SHA256(42)
    x = prng.random(10)
    y = x
    group = prng.randint(0, 3, size=10)
    res1 = sim_corr(x, y, group, seed=prng, reps=100)
    res2 = sim_corr(x, y, group, seed=prng, alternative='less', reps=100)
    res3 = sim_corr(x, y, group, seed=prng, alternative='two-sided', reps=100)
    
    np.testing.assert_almost_equal(res1[0], 1-res2[0])
    assert res1[1] == res2[1]
    assert res1[1] == res3[1]
    assert 2*res1[0] == res3[0]


def test_strat_tests_equal():
    group = np.repeat([1, 2, 3], 10)
    condition = np.repeat([1, 2] * 3, 5)
    response = np.zeros_like(group)
    response[[0, 1, 3, 9, 10, 11, 18, 19, 20]] = 1

    res1 = spt(group, condition, response, reps=100, seed=42)
    res2 = stratified_two_sample(group, condition, response, reps=100,
                                stat='mean_within_strata', seed=42)
    assert res1[1] == res2[1]
    assert math.fabs(res1[0]-res2[0]) < 0.05
    
def test_stratified_two_sample():
    group = np.repeat([1, 2, 3], 10)
    condition = np.repeat([1, 2] * 3, 5)
    response = np.zeros_like(group)
    response[[0, 1, 3, 9, 10, 11, 18, 19, 20]] = 1

    res = stratified_two_sample(group, condition, response, reps=1000,
                                stat='mean', seed=42)
    np.testing.assert_almost_equal(res[0], 0.245, 2)
    assert res[1] == 0.2
    
    (p, t, dist) = stratified_two_sample(group, condition, response, reps=1000,
                                stat='mean', seed=42, keep_dist=True)
    assert res == (p, t)
    
    stat_fun = lambda u: sptm(group, condition, u, np.unique(group), np.unique(condition))
    res = stratified_two_sample(group, condition, response, reps=100,
                                stat=stat_fun, seed=42)
    np.testing.assert_almost_equal(res[0], 0.8712, 3)
    assert res[1] == 0.30
