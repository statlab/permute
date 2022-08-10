# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:23:07 2022

@author: Clayton
"""

import pytest

import numpy as np
from numpy.random import RandomState
from cryptorandom.cryptorandom import SHA256

from ..multitest_core import (multitest_corr,
                    multitest_spearman_corr,
                    multitest_two_sample,
                    multitest_two_sample_shift,
                    multitest_two_sample_conf_int,
                    multitest_one_sample)



def multitest_test_corr():
    prng = SHA256(42)
    num_tests = 2
    x = prng.randint(0, 5, size=(10,num_tests))
    y = x
    res1 = multitest_corr(x, y, seed=prng)
    res2 = multitest_corr(x, y)
    assert len(res1) == 3
    assert len(res2) == 3
    np.testing.assert_almost_equal(res1[0], np.ones(num_tests), decimal=1)
    np.testing.assert_almost_equal(res2[0], np.ones(num_tests), decimal=1)
    np.testing.assert_almost_equal(res1[1], res2[1], decimal=1)
    print("finished test 1 in test_corr()")
    
    y = prng.randint(0, 5, size=(10,num_tests))
    res1 = multitest_corr(x, y, alternative="less", seed=prng)
    res2 = multitest_corr(x, y, alternative="less")
    assert len(res1) == 3
    assert len(res2) == 3
    np.testing.assert_almost_equal(res1[0], res2[0], decimal=1)
    np.testing.assert_almost_equal(res1[1], res2[1], decimal=1)
    print("finished test 2 in test_corr()")
    
    res1 = multitest_corr(x, y, alternative="two-sided", seed=prng)
    res2 = multitest_corr(x, y, alternative="greater")
    assert len(res1) == 3
    assert len(res2) == 3
    np.testing.assert_almost_equal(res1[0], res2[0], decimal=1)
    np.testing.assert_almost_equal(res1[1], res2[1]*2, decimal=1)
    print("finished test 3 in test_corr()")


def multitest_test_spearman_corr():
    prng = SHA256(42)
    num_tests = 2
    x = (np.array([2, 4, 6, 8, 10])*np.ones((num_tests,5))).T
    y = (np.array([1, 3, 5, 6, 9])*np.ones((num_tests,5))).T
    xorder = np.array([[1, 2, 3, 4, 5] ,[1, 2, 3, 4, 5]]).T
    res1 = multitest_corr(xorder, xorder, seed=prng)
    print("finished test 1 in test_spearman_corr()")
        
    prng = SHA256(42)
    res2 = multitest_spearman_corr(x, y, seed=prng)
    np.testing.assert_almost_equal(res1[0], res2[0], decimal=1)
    np.testing.assert_almost_equal(res1[1], res2[1], decimal=1)
    np.testing.assert_array_equal(res1[2], res2[2])
    print("finished test 2 in test_spearman_corr()")


@pytest.mark.slow
def multitest_test_two_sample():
    prng = RandomState(42)
    num_samples = 200
    num_tests = 2
    # Normal-normal, different means examples
    x = prng.normal(1, size=(num_samples,num_tests))
    y = prng.normal(4, size=(num_samples,num_tests))
    expected = ([[1.0, 1.0],[0,0]], [[-3,-3],[-30,-30]]) # define expected probabilites 
    plus1 = (True,False)
    keep_dist = (True,False)
    alternative = ('greater','less','two-sided')
    stats = ('mean','t') # TODO add custom stat lambda here
    num_cases = str(len(plus1)*len(keep_dist)*len(alternative)*len(stats))
    case_count = 1
    # go through all combinations of parameters
    for p in plus1:
        for k in keep_dist:
            for a in alternative:
                for s in stats:
                    res = multitest_two_sample(x,y,reps=10**4,seed=42,plus1=p,keep_dist=k,alternative=a,stat=s)
                    # check pvals
                    if a == 'greater':
                        np.testing.assert_almost_equal(res[0], expected[0][0], decimal = 1) # compare p vals
                    elif a == 'less' or a =='two-sided':
                        np.testing.assert_almost_equal(res[0], expected[0][1], decimal = 1) # compare p vals
                    # check observed statistic
                    if s == 'mean':
                        np.testing.assert_almost_equal(res[1], expected[1][0], decimal = 1) # compare observed statistic
                    elif s == 't':
                        np.testing.assert_almost_equal(res[1], expected[1][1], decimal = 0) # compare observed statistic
                    #check returned distribution
                    if k:
                        assert len(res) == 3 # if keep keep dist, expect to res to be length 3
                        assert len(res[2].shape) == 2 # should get 2D dist
                        assert res[2].shape[1] == num_tests # second D should have same number of elements as number of tests
                    else:
                        assert len(res) == 2
                    print("finished test " + str(case_count) + " of " + num_cases + " in test_two_sample()")
                    case_count += 1
    
@pytest.mark.slow
def multitest_test_one_sample():
    # same code as multitest_test_two_sample(), but switched tested function to one sample version
    prng = RandomState(42)
    num_samples = 200
    num_tests = 2
    # Normal-normal, different means examples
    x = prng.normal(1, size=(num_samples,num_tests))
    y = prng.normal(4, size=(num_samples,num_tests))
    expected = ([[1.0, 1.0],[0,0]], [[-3,-3],[-28,-28]]) # define expected probabilites for different alternative hypotheses and observed statistics (not sure why observed statistic is different for one and two sample tests)
    plus1 = (True,False)
    keep_dist = (True,False)
    alternative = ('greater','less','two-sided')
    stats = ('mean','t') # TODO add custom stat lambda here
    num_cases = str(len(plus1)*len(keep_dist)*len(alternative)*len(stats))
    case_count = 1
    # go through all combinations of parameters
    for p in plus1:
        for k in keep_dist:
            for a in alternative:
                for s in stats:
                    res = multitest_one_sample(x,y,seed=42,reps=10**4,plus1=p,keep_dist=k,alternative=a,stat=s)
                    # check pvals
                    if a == 'greater':
                        np.testing.assert_almost_equal(res[0], expected[0][0], decimal = 1) # compare p vals
                    elif a == 'less' or a =='two-sided':
                        np.testing.assert_almost_equal(res[0], expected[0][1], decimal = 1) # compare p vals
                    # check observed statistic
                    if s == 'mean':
                        np.testing.assert_almost_equal(res[1], expected[1][0], decimal = 1) # compare observed statistic
                    elif s == 't':
                        np.testing.assert_almost_equal(res[1], expected[1][1], decimal = 0) # compare observed statistic
                    #check returned distribution
                    if k:
                        assert len(res) == 3 # if keep keep dist, expect to res to be length 3
                        assert len(res[2].shape) == 2 # should get 2D dist
                        assert res[2].shape[1] == num_tests # second D should have same number of elements as number of tests
                    else:
                        assert len(res) == 2
                    print("finished test " + str(case_count) + " of " + num_cases + " in test_one_sample()")
                    case_count += 1


# @pytest.mark.slow
# def multitest_test_two_sample_shift():
#     f = lambda u: u - 3
#     finv = lambda u: u + 3
#     f_err = lambda u: 2 * u
#     f_err_inv = lambda u: u / 2
#     prng = RandomState(42)
#     num_samples = 200
#     num_tests = 2
#     # Normal-normal, different means examples
#     x = prng.normal(1, size=(num_samples,num_tests))
#     y = prng.normal(4, size=(num_samples,num_tests))
#     expected = ([[1.0, 1.0],[0,0]], [[-3,-3],[-28,-28]]) # define expected probabilites for different alternative hypotheses and observed statistics (not sure why observed statistic is different for one and two sample tests)
#     plus1 = (True,False)
#     max_correct = (True,False)
#     keep_dist = (True,False)
#     alternative = ('greater','less','two-sided')
#     stats = ('mean','t') # TODO add custom stat lambda here
#     shift = (2,(f, finv),(f_err, f_err_inv))
#     num_cases = str(len(plus1)*len(keep_dist)*len(alternative)*len(stats)*len(shift))
#     case_count = 1
#     # go through all combinations of parameters
#     for p in plus1:
#         for k in keep_dist:
#             for a in alternative:
#                 for s in stats:
#                     for sh in shift:
#                         res = multitest_two_sample_shift(x,y,seed=42,shift = sh, plus1=p,keep_dist=k,alternative=a,stat=s)
#                         # check pvals
#                         if a == 'greater':
#                             np.testing.assert_almost_equal(res[0], expected[0][0], decimal = 1) # compare p vals
#                         elif a == 'less' or a =='two-sided':
#                             np.testing.assert_almost_equal(res[0], expected[0][1], decimal = 1) # compare p vals
#                         # check observed statistic
#                         if s == 'mean':
#                             np.testing.assert_almost_equal(res[1], expected[1][0], decimal = 1) # compare observed statistic
#                         elif s == 't':
#                             np.testing.assert_almost_equal(res[1], expected[1][1], decimal = 0) # compare observed statistic
#                         #check returned distribution
#                         if k:
#                             assert len(res[2].shape) == 2 # if not max correct, should get 2D dist
#                             assert res[2].shape[1] == num_tests # second D should have same number of elements as number of tests
#                         else:
#                             assert len(res) == 2
#                         print("finished test " + str(case_count) + " of " + num_cases + " in test_one_sample()")
#                         case_count += 1


# def test_two_sample_bad_shift():
#     # Break it with a bad shift
#     x = np.array(range(5))
#     y = np.array(range(1, 6))
#     shift = lambda u: u + 3
#     pytest.raises(ValueError, multitest_two_sample_shift, x, y, seed=5, shift=shift)

