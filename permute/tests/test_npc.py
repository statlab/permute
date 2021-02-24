import pytest

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm

from ..npc import (fisher,
                   liptak,
                   tippett,
                   inverse_n_weight,
                   npc,
                   check_combfunc_monotonic,
                   sim_npc,
                   fwer_minp,
                   randomize_group,
                   Experiment)


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


def test_npc():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=500).reshape(100, 5)
    res = npc(pvalues, distr, "fisher", plus1=False)
    np.testing.assert_almost_equal(res, 0.33)
    res = npc(pvalues, distr, "liptak", plus1=False)
    np.testing.assert_almost_equal(res, 0.35)
    res = npc(pvalues, distr, "tippett", plus1=False)
    np.testing.assert_almost_equal(res, 0.25)


def test_npc_callable_combine():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=500).reshape(100, 5)
    size = np.array([2, 4, 6, 4, 2])
    combine = lambda p: inverse_n_weight(p, size)
    res = npc(pvalues, distr, combine, plus1=False)
    np.testing.assert_equal(res, 0.39)


def test_npc_bad_distr():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=20).reshape(10, 2)
    pytest.raises(ValueError, npc, pvalues, distr, "fisher")


def test_npc_single_pvalue():
    pytest.raises(ValueError, npc, np.array([1]), np.array([1, 2, 3]))
    

def test_monotonic_checker():
    pvalues = np.array([0.1, 0.2, 0.3])
    np.testing.assert_equal(check_combfunc_monotonic(pvalues, fisher), True)
    np.testing.assert_equal(check_combfunc_monotonic(pvalues, liptak), True)
    np.testing.assert_equal(check_combfunc_monotonic(pvalues, tippett), True)
    
    comb_function = lambda p: inverse_n_weight(p, np.array([2, 4, 6]))
    np.testing.assert_equal(check_combfunc_monotonic(pvalues, comb_function), True)
    
    bad_comb_function = lambda p: -1*fisher(p)
    np.testing.assert_equal(check_combfunc_monotonic(pvalues, bad_comb_function), False)
    

def test_mono_checker_in_npc():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=500).reshape(100, 5)
    bad_comb_function = lambda p: -1*fisher(p)
    pytest.raises(ValueError, npc, pvalues, distr, bad_comb_function)


def test_minp_bad_distr():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=20).reshape(10, 2)
    pytest.raises(ValueError, fwer_minp, pvalues, distr, "fisher")    
    

def test_minp_one_pvalue():
    prng = RandomState(55)
    pvalues = np.array([1])
    distr = prng.uniform(low=0, high=10, size=20).reshape(20, 1)
    pytest.raises(ValueError, fwer_minp, pvalues, distr, "fisher")


def test_sim_npc():
    prng = RandomState(55)
    # test Y always greater than X so p-value should be 1 
    responses = np.array([[0, 1], [0, 1], [1, 2], [1, 2]])
    group = np.array([1, 1, 2, 2])
    my_randomizer = Experiment.Randomizer(randomize = randomize_group, seed = prng)
    data = Experiment(group, responses)
    
    # create median test statistic to apply to every column
    def med_diff(data, resp_index):
        # get response variable for that index
        resp = np.array([item[resp_index] for item in data.response])
        # get unique groups
        groups = np.unique(data.group)
        # get mean for each group
        mx = np.nanmean(resp[data.group == groups[0]])
        my = np.nanmean(resp[data.group == groups[1]])
        return mx-my
    
    test_array = Experiment.make_test_array(med_diff, [0, 1])
    res = sim_npc(data, test_array, combine="fisher", seed=None, reps=int(1000))
    np.testing.assert_almost_equal(res[0], 1)
    
    # test X = Y so p-value should be 1
    responses = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
    group = np.array([1, 1, 2, 2])
    data = Experiment(group, responses, randomizer = my_randomizer)
    res = sim_npc(data, test = Experiment.make_test_array(Experiment.TestFunc.mean_diff, [0, 1]), 
                  combine="fisher", seed=None, reps=int(1000))
    np.testing.assert_almost_equal(res[0], 1)
    
    # test stat for cat_1 is smaller if X all 0s which about 0.015 chance so pvalue should be about 0.985
    responses = np.array([[0, 1], [1, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 1], [0, 1]])
    group = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    data = Experiment(group, responses, randomizer = my_randomizer)
    res = sim_npc(data, test = Experiment.make_test_array(Experiment.TestFunc.mean_diff, [0, 1]),
                  combine="fisher", seed=None, reps=int(1000))
    np.testing.assert_almost_equal(res[0], 0.985, decimal = 2)
    
    
def test_fwer_minp():
    prng = RandomState(55)
    pvalues = np.linspace(0.05, 0.9, num=5)
    distr = prng.uniform(low=0, high=10, size=100000).reshape(20000, 5)
    res = fwer_minp(pvalues, distr, "fisher", plus1=False)
    expected_res = np.array([0.348594, 0.744245, 0.874132, 0.915783, 0.915783])
    np.testing.assert_almost_equal(res, expected_res, decimal=2)
    res = fwer_minp(pvalues, distr, "tippett", plus1=False)
    expected_res = np.array([0.2262191, 0.704166, 0.8552969, 0.9023438, 0.9023438])
    np.testing.assert_almost_equal(res, expected_res, decimal=2)
