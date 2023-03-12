"""
Unit Tests for stratified.py
"""
import numpy as np
import math
from numpy.random import RandomState

import pytest
from cryptorandom.cryptorandom import SHA256

from ..stratified import stratified_permutationtest
from ..stratified import stratified_permutationtest_mean
from ..stratified import corrcoef, sim_corr, stratified_two_sample

def test_corrcoef():
    assert corrcoef(np.array([1,2,3]),np.array([3,4,5]),np.array([1,1,1]))==1.0
    assert corrcoef(np.array([3,2,1]),np.array([3,4,5]),np.array([1,1,1]))==-1.0
    
def test_sim_corr():
    assert sim_corr(np.array([1,2,3]),np.array([3,4,5]),np.array([1,1,1]))[1]==1.0