"""
Unit Tests for utils.py
"""
import sys
import pytest
import math
import numpy as np
from scipy.optimize import brentq
from scipy.stats import hypergeom, binom
from cryptorandom.cryptorandom import SHA256
from cryptorandom.sample import random_sample, random_permutation

from ..utils import (binom_conf_interval,
                     hypergeom_conf_interval,
                     hypergeometric,
                     binomial_p,
                     get_prng,
                     permute,
                     permute_rows,
                     permute_within_groups,
                     permute_incidence_fixed_sums,
                     potential_outcomes)


def test_binom_conf_interval():
    obs1 = binom_conf_interval(15,5)
    expected1 = (0.09957627246482563, 0.649793470222527)
    np.testing.assert_almost_equal(obs1, expected1, decimal=3)
    
    obs2 = binom_conf_interval(15,9,cl=0.9)
    expected2 = (0.3595652102176309, 0.8091353138258284)
    np.testing.assert_almost_equal(obs2, expected2, decimal=3)
    
    obs3 = binom_conf_interval(15,13,cl=0.95,alternative='lower')
    expected3 = (0.636558234458598, 1.0)
    np.testing.assert_almost_equal(obs3, expected3, decimal=3)
    
    obs4 = binom_conf_interval(15,1,cl=0.95,alternative='upper')
    expected4 = (0.0, 0.2793961936129413)
    np.testing.assert_almost_equal(obs4, expected4, decimal=3)
    

def test_hypergeom_conf_interval():
    obs1 = hypergeom_conf_interval(5,3,10)
    expected1 = (3.0, 8.0)
    np.testing.assert_equal(obs1, expected1)
    
    obs2 = hypergeom_conf_interval(5,3,10,cl=0.6)
    expected2 = (4.0, 7.0)
    np.testing.assert_equal(obs2, expected2)
    
    obs3 = hypergeom_conf_interval(6,1,10,cl=0.95,alternative='upper')
    expected3 = (0.0, 4.0)
    np.testing.assert_equal(obs3, expected3)
    
    obs4 = hypergeom_conf_interval(4,3,10,cl=0.95,alternative='lower')
    expected4 = (4.0, 10.0)
    np.testing.assert_equal(obs4, expected4)
    
def test_hypergeometric():
    obs1 = hypergeometric(10,10,10,10)
    expected1 = 1.0
    np.testing.assert_equal(obs1, expected1)
    
    obs2 = hypergeometric(9,10,9,9,alternative='greater')
    expected2 = 0.1
    np.testing.assert_almost_equal(obs2, expected2, decimal=3)
    
    obs3 = hypergeometric(0,10,9,1,alternative='less')
    expected3 = 0.1
    np.testing.assert_almost_equal(obs3, expected3, decimal=3)
    
    obs4 = hypergeometric(0,10,9,1,alternative='two-sided')
    expected4 = 0.2
    np.testing.assert_almost_equal(obs4, expected4, decimal=3)
    
def test_binomial_p():
    obs1 = binomial_p(10,10,0)
    expected1 = 0.0
    np.testing.assert_equal(obs1, expected1)
    
    obs2 = binomial_p(10,10,1,alternative='two-sided')
    expected2 = 1.0
    np.testing.assert_equal(obs2, expected2)
    
    obs3 = binomial_p(0,10,1,alternative='less')
    expected3 = 0.0
    np.testing.assert_equal(obs3, expected3)
    
def test_get_prng():
    x1 = get_prng()
    obs = x1.randint(0,10,size=20)
    assert len(obs)==20
    for i in np.arange(len(obs)):
        assert obs[i]<10 and obs[i]>=0

def test_permute_within_groups():
    obs1 = np.sort(permute_within_groups(np.array([1,2,3]),np.array([3,3,3])))
    expected1 = np.array([1,2,3])
    assert np.array_equal(obs1,expected1)
    
    obs2 = permute_within_groups(np.array([1,2,3]),np.array([1,2,3]))
    expected2 = np.array([1,2,3])
    assert np.array_equal(obs2,expected2)

def test_permute():
    obs1 = np.sort(permute(np.array([4,7,8,3])))
    expected1 = np.array([3,4,7,8])
    assert np.array_equal(obs1,expected1)
    
def test_permute_rows():
    obs_array = permute_rows(np.array([[1,2,3],[2,3,4]]))
    obs1 = np.sort(obs_array[0,])
    expected1 = np.array([1,2,3])
    assert np.array_equal(obs1,expected1)
    
    obs2 = np.sort(obs_array[1,])
    expected2 = np.array([2,3,4])
    assert np.array_equal(obs2,expected2)