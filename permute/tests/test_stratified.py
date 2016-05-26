from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import math
from numpy.random import RandomState

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr

from ..stratified import stratified_permutationtest as spt
from ..stratified import stratified_permutationtest_mean as sptm
from ..stratified import corrcoef, sim_corr


def test_stratified_permutationtest():
    group = np.repeat([1, 2, 3], 9)
    condition = np.repeat([1, 2, 3] * 3, 3)
    response = np.zeros_like(group)
    response[[0, 1, 3, 9, 10, 11, 18, 19, 20]] = 1

    res = spt(group, condition, response, reps=1000, seed=42)
    res1 = spt(group, condition, response, alternative='less', reps=1000, seed=42)
    assert_less(res[0], 0.01)
    assert_equal(res[1], res1[1])
    assert_almost_equal(res[0], 1-res1[0])
    res2 = spt(group, condition, response, alternative='two-sided', reps=1000, seed=42)
    assert_less(res2[0], 0.02)

    group = np.array([1, 1, 1])
    condition = np.array([2, 2, 2])
    response = np.zeros_like(group)
    res2 = spt(group, condition, response, reps=1000, seed=42)
    assert_equal(res2, (1.0, np.nan, None))


def test_stratified_permutationtest_mean():
    group = np.array([1, 2, 1, 2])
    condition = np.array([1, 1, 2, 2])
    response = np.zeros_like(group)
    groups = np.unique(group)
    conditions = np.unique(condition)
    res = sptm(group, condition, response, groups, conditions)
    assert_equal(res, 0.0)
    res2 = sptm(group, condition, response)  # check defaults work
    assert_equal(res2, 0.0)


@raises(ValueError)
def test_stratified_permutationtest_mean_error():
    group = np.array([1, 1, 1])
    condition = np.array([2, 2, 2])
    response = np.zeros_like(group)
    groups = np.unique(group)
    conditions = np.unique(condition)
    res = sptm(group, condition, response, groups, conditions)


def test_corrcoef():
    prng = RandomState(42)
    x = prng.rand(10)
    y = x
    group = prng.randint(3, size=10)
    res1 = corrcoef(x, y, group)
    res2 = corrcoef(x, y, group)
    assert_equal(res1, res2)


#@attr('slow')
def test_sim_corr():
    prng = RandomState(42)
    x = prng.rand(10)
    y = x
    group = prng.randint(3, size=10)
    res1 = sim_corr(x, y, group, seed=prng, reps=100)
    res2 = sim_corr(x, y, group, seed=prng, alternative='less', reps=100)
    res3 = sim_corr(x, y, group, seed=prng, alternative='two-sided', reps=100)
    
    assert_almost_equal(res1[0], 1-res2[0])
    assert_equal(res1[1], res2[1])
    assert_equal(res1[1], res3[1])
    assert_equal(res1[0], res3[0])
