from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from nose.plugins.attrib import attr
from nose.tools import assert_raises, raises

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm

from ..npc import (fisher,
                   liptak,
                   tippett,
                   inverse_n_weight,
                   t2p,
                   npc,
                   check_combfunc_monotonic,
                   fwer_minp)


def test_fisher():
    pvalues = np.linspace(0.05, 0.9, num=5)
    np.testing.assert_almost_equal(fisher(pvalues), 11.11546, 5)
    np.testing.assert_equal(fisher(1), -0.0)
    np.testing.assert_array_less(fisher(10), 0)


def test_liptak():
    pvalues = np.linspace(0.05, 0.9, num=5)
    np.testing.assert_almost_equal(liptak(pvalues), 0.5728894, 5)
    np.testing.assert_equal(liptak(1), norm.ppf(0))
    np.testing.assert_equal(liptak(10), np.nan)


def test_tippett():
    pvalues = np.linspace(0.05, 0.9, num=5)
    np.testing.assert_almost_equal(tippett(pvalues), 0.95, 5)
    np.testing.assert_equal(tippett(1), 0)
    np.testing.assert_equal(tippett(10), -9)


def test_inverse_n_weight():
    pval = np.array([0.5, 0.25, 0.75])
    size = np.array([2, 4, 6])
    expected_npc = -0.7847396
    res_npc = inverse_n_weight(pval, size)
    np.testing.assert_almost_equal(expected_npc, res_npc)


def test_t2p():
    obs = 5
    distr = np.array(range(-10, 11))
    expected = np.linspace(21, 1, num=21)/21
    np.testing.assert_array_almost_equal(t2p(distr, "greater", plus1=False), expected)
    np.testing.assert_array_almost_equal(t2p(distr, "less", plus1=False), expected[::-1])
    
    expected2 = 2*np.concatenate([expected[::-1][:10], 
        [0.5], expected[11:]])
    np.testing.assert_array_almost_equal(t2p(distr, "two-sided", plus1=False), expected2)


@raises(ValueError)
def test_t2p_bad_alternative():
    t2p(np.array([0.5, 0.25, 0.75]), "not a real alternative")


def test_npc():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=500).reshape(100, 5)
    res = npc(pvalues, distr, "fisher", "greater", plus1=False)
    np.testing.assert_almost_equal(res, 0.33)
    res = npc(pvalues, distr, "fisher", "less", plus1=False)
    np.testing.assert_almost_equal(res, 0.33)
    res = npc(pvalues, distr, "fisher", "two-sided", plus1=False)
    np.testing.assert_almost_equal(res, 0.31)
    res = npc(pvalues, distr, "liptak", "greater", plus1=False)
    np.testing.assert_almost_equal(res, 0.35)
    res = npc(pvalues, distr, "tippett", "greater", plus1=False)
    np.testing.assert_almost_equal(res, 0.25)
    res = npc(pvalues, distr, "fisher",
              alternatives=np.array(["less", "greater", "less",
                                     "greater", "two-sided"]), plus1=False)
    np.testing.assert_almost_equal(res, 0.38)


def test_npc_callable_combine():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=500).reshape(100, 5)
    size = np.array([2, 4, 6, 4, 2])
    combine = lambda p: inverse_n_weight(p, size)
    res = npc(pvalues, distr, combine, "greater", plus1=False)
    np.testing.assert_equal(res, 0.39)


@raises(ValueError)
def test_npc_bad_distr():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=20).reshape(10, 2)
    npc(pvalues, distr, "fisher", "greater")


@raises(ValueError)
def test_npc_bad_alternative():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=50).reshape(10, 5)
    npc(pvalues, distr, "fisher", np.array(["greater", "less"]))


@raises(ValueError)
def test_npc_single_pvalue():
    npc(np.array([1]), np.array([1, 2, 3]))
    

def test_monotonic_checker():
    pvalues = np.array([0.1, 0.2, 0.3])
    np.testing.assert_equal(check_combfunc_monotonic(pvalues, fisher), True)
    np.testing.assert_equal(check_combfunc_monotonic(pvalues, liptak), True)
    np.testing.assert_equal(check_combfunc_monotonic(pvalues, tippett), True)
    
    comb_function = lambda p: inverse_n_weight(p, np.array([2, 4, 6]))
    np.testing.assert_equal(check_combfunc_monotonic(pvalues, comb_function), True)
    
    bad_comb_function = lambda p: -1*fisher(p)
    np.testing.assert_equal(check_combfunc_monotonic(pvalues, bad_comb_function), False)
    

@raises(ValueError)
def test_mono_checker_in_npc():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=500).reshape(100, 5)
    bad_comb_function = lambda p: -1*fisher(p)
    npc(pvalues, distr, bad_comb_function)


@raises(ValueError)
def test_minp_bad_distr():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=20).reshape(10, 2)
    fwer_minp(pvalues, distr, "fisher", "greater")


@raises(ValueError)
def test_minp_one_pvalue():
    prng = RandomState(55)
    pvalues = np.array([1])
    distr = prng.uniform(low=0, high=10, size=20).reshape(20, 1)
    npc(pvalues, distr, "fisher", "greater")

# TODO: more fwer_minp tests