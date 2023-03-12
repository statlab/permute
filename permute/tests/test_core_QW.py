"""
Unit Tests for core.py
"""

import pytest

import numpy as np
from numpy.random import RandomState
from cryptorandom.cryptorandom import SHA256

from ..core import (corr,
                    spearman_corr,
                    two_sample,
                    two_sample_shift,
                    two_sample_conf_int,
                    one_sample)

def test_corr():
    obs = corr(np.array([1,2,3]),np.array([1,2,3]))
    assert obs[0]==1.0
    
def test_spearman_corr():
    obs = spearman_corr(np.array([1,2,3]),np.array([1,2,3]))
    assert obs[0]==1.0
    
def test_two_sample():
    obs1 = two_sample(np.array([1,2,3]),np.array([4,5,6]))
    assert obs1[0]==1.0
    assert obs1[1]==-3.0
    
    obs2 = two_sample(np.array([7,8,9]),np.array([1,2,3]),alternative='less')
    assert obs2[0]==1.0
    assert obs2[1]==6.0
    
    obs3 = two_sample(np.array([2,2,2]),np.array([2,2,2]),alternative='two-sided')
    assert obs3[0]==1.0
    assert obs3[1]==0.0
    
def test_two_sample_shift():
    obs1 = two_sample_shift(np.array([1,2,3]),np.array([4,5,6]),shift=1)
    assert obs1[0]==1.0
    assert obs1[1]==-3.0
    
    obs2 = two_sample_shift(np.array([7,8,9]),np.array([1,2,3]),alternative='less',shift=1)
    assert obs2[0]==1.0
    assert obs2[1]==6.0
    
    obs3 = two_sample_shift(np.array([2,2,2]),np.array([2,2,2]),alternative='two-sided',shift=0)
    assert obs3[0]==1.0
    assert obs3[1]==0.0
    
def test_one_sample():
    obs1 = one_sample(np.array([0,0,0,0,0]))
    assert obs1[0]==1.0
    assert obs1[1]==0.0
    
    obs2 = one_sample(np.array([10,10,10,10,10]),alternative='less')
    assert obs2[0]==1.0
    assert obs2[1]==10.0
    
    obs3 = one_sample(np.array([0,0,0,0,0]),alternative='two-sided')
    assert obs3[0]==1.0
    assert obs3[1]==0.0
    