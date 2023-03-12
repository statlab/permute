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
                   westfall_young,
                   adjust_p,
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


def test_westfall_young():
    prng = RandomState(55)
    # test from https://support.sas.com/kb/22/addl/fusion22950_1_multtest.pdf
    group = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    responses = np.array([[14.4, 7.00, 4.30], [14.6, 7.09, 3.88 ], [13.8, 7.06, 5.34], [10.1, 4.26, 4.26], [11.1, 5.49, 4.52], [12.4, 6.13, 5.69], [12.7, 6.69, 4.45], [11.8, 5.44, 3.94], [18.3, 1.28, 0.67], [18.0, 1.50, 0.67], [20.8, 1.51, 0.72], [18.3, 1.14, 0.67], [14.8, 2.74, 0.67], [13.8, 7.08, 3.43], [11.5, 6.37, 5.64], [10.9, 6.26, 3.47]])
    data = Experiment(group, responses)
    test = Experiment.make_test_array(Experiment.TestFunc.ttest, [0, 1, 2])
    result = westfall_young(data, test, method = "minP", alternatives = 'two-sided', seed = prng) 
    np.testing.assert_almost_equal(result[0][0], 0.1, decimal = 1)
    np.testing.assert_almost_equal(result[0][1], 0.05, decimal = 2)
    np.testing.assert_almost_equal(result[0][2], 0.02, decimal = 2)
    np.testing.assert_almost_equal(result[1][0], 0.1, decimal = 1)
    np.testing.assert_almost_equal(result[1][1], 0.03, decimal = 2)
    np.testing.assert_almost_equal(result[1][2], 0.01, decimal = 2)
    result = westfall_young(data, test, method = "maxT", alternatives = 'two-sided')
    np.testing.assert_almost_equal(result[0][1], 0.05, decimal = 2)
    np.testing.assert_almost_equal(result[0][2], 0.02, decimal = 2)
    result  = westfall_young(data, test, method = "minP", alternatives = 'greater')
    np.testing.assert_almost_equal(result[0][1], 0.03, decimal = 2)
    np.testing.assert_almost_equal(result[1][2], 0.005, decimal = 3)
    result  = westfall_young(data, test, method = "maxT", alternatives = 'greater')
    np.testing.assert_almost_equal(result[0][0], 1.0, decimal = 1)
    np.testing.assert_almost_equal(result[1][0], 0.95, decimal = 2)
    
    
    
def test_adjust_p():
    pvalues = np.array([0.1, 0.2, 0.3, 0.4])
    # bonferroni
    res = adjust_p(pvalues, adjustment = 'bonferroni')
    np.testing.assert_almost_equal(res, np.array([0.4, 0.8, 1, 1]), decimal = 2)
    # holm-bonferroni
    pvalues = np.array([0.01, 0.04, 0.03, 0.005])                     
    res = adjust_p(pvalues, adjustment = 'holm-bonferroni')
    np.testing.assert_almost_equal(res, np.array([0.03, 0.06, 0.06, 0.02]), decimal = 2)
    # benjamini hochberg
    res = adjust_p(pvalues, adjustment = 'benjamini-hochberg')
    np.testing.assert_almost_equal(res, np.array([0.02, 0.04, 0.04, 0.02]), decimal = 2)
    # raises value error for nonsense correction
    pytest.raises(ValueError, adjust_p, pvalues, 'nonsense')
    
                            
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
