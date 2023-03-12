"""
Unit Tests for ksample.py
"""

import numpy as np
from scipy.stats import hypergeom, binom
from cryptorandom.cryptorandom import SHA256

from ..ksample import (k_sample, 
                       one_way_anova,
                       bivariate_k_sample,
                       two_way_anova)
import permute.data as data
from permute.utils import get_prng

def test_k_sample():
    obs = k_sample(np.array([1,1,1,2,2,2]),np.array([1,1,1,1,1,1]))
    assert obs[0]==1.0
    assert obs[1]==0.0
    
def test_one_way_anova():
    assert one_way_anova(np.array([1,1,1,2,2,2]),np.array([1,1,1,1,1,1]),np.mean(np.array([1,1,1,2,2,2])))==0.0
    
def test_bivariate_k_sample():
    obs = bivariate_k_sample(np.array([1,1,1,2,2,2]),np.array([1,1,1,1,1,1]),np.array([1,1,1,1,1,1]))
    assert obs[0]==1.0
    assert obs[1]==0.0
    
def test_two_way_anova():
    two_way_anova(np.array([1,1,1,2,2,2]),np.array([1,1,1,1,1,1]),np.array([1,1,1,1,1,1]),np.mean(np.array([1,1,1,2,2,2])))==0.0