import numpy as np
from scipy.stats import hypergeom, binom
from cryptorandom.cryptorandom import SHA256

from ..ksample import (k_sample, 
                       one_way_anova,
                       bivariate_k_sample,
                       two_way_anova)
import permute.data as data
from permute.utils import get_prng


def test_worms_ksample():
    worms = data.worms()
    res = k_sample(worms.x, worms.y, stat='one-way anova', reps=1000, seed=1234)
    np.testing.assert_array_less(0.006, res[0])
    np.testing.assert_array_less(res[0], 0.02)


def test_one_way_anova():
    group = np.ones(5)
    x = np.array(range(5))
    xbar = np.mean(x)
    assert one_way_anova(x, group, xbar) == 0
    
    group = np.array([1]*3 + [2]*2)
    expected = 3*1**2 + 2*1.5**2
    assert one_way_anova(x, group, xbar) == expected


def test_two_way_anova():
    prng = get_prng(100)
    group1 = np.array([1]*5 + [2]*5)                                              
    group2 = np.array(list(range(5))*2)                                           
    x = prng.randint(1, 10, 10)
    xbar = np.mean(x)
    val = two_way_anova(x, group1, group2, xbar)
    np.testing.assert_almost_equal(val, 0.296, 3)
    
    x = group2 + 1
    xbar = 3
    assert two_way_anova(x, group1, group2, xbar) == 1


def test_testosterone_ksample():
    testosterone = data.testosterone()
    x = np.hstack(testosterone.tolist())
    group1 = np.hstack([[i]*5 for i in range(len(testosterone))])
    group2 = np.array(list(range(5))*len(testosterone))
    assert len(group1) == 55
    assert len(group2) == 55
    assert len(x) == 55
    res = bivariate_k_sample(x, group1, group2, reps=5000, seed=5)
    np.testing.assert_array_less(res[0], 0.0002)


import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import f_oneway

def test_bivariate_k_sample_keep_dist_true():
    np.random.seed(123)
    x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    group1 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    group2 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    reps = 1000
    pval, tst, dist = bivariate_k_sample(x, group1, group2, reps=reps, stat='two-way anova', keep_dist=True)

    # check that output has correct dimensions and types
    assert isinstance(pval, float)
    assert isinstance(tst, float)
    assert isinstance(dist, np.ndarray)
    assert pval >= 0 and pval <= 1
    assert tst >= 0
    assert dist.shape == (reps,)

    # compare 
    np.testing.assert_array_less(pval, 0.05)
