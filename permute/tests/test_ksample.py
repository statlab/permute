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

def test_ksample_abs_mean_diff():
    """Test whether two groups are the same, using the absolute difference between means of the groups
    as the test statistic. The treatment group has a higher mean than the control group, and the given
    group assignment should have the largest test statistic compared to any permutation of group labels.
    """
    x = np.array(range(10))
    group_labels = np.zeros_like(x)
    group_labels[-4:] = 1
    def abs_diff_two_mean_stat(x, group, overall_mean):
        x = np.array(x)
        tst = 0
        assert set(group) == set([0, 1])
        return np.abs(np.mean(x[group == 0]) - np.mean(x[group == 1]))
    pval, test_stat, dist = k_sample(x, group_labels, stat=abs_diff_two_mean_stat, keep_dist=True)
    max_ts = abs_diff_two_mean_stat(x, group_labels, np.mean(x))
    assert np.isclose(np.max(dist), max_ts)


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
