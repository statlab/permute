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
    prng = SHA256(42)
    x = prng.randint(0, 5, size=10)
    y = x
    res1 = corr(x, y, seed=prng)
    res2 = corr(x, y)
    assert len(res1) == 3
    assert len(res2) == 3
    assert res1[0] == 1
    assert res2[0] == 1
    np.testing.assert_almost_equal(res1[1], res2[1], decimal=1)
    print("finished test 1 in test_corr()")

    y = prng.randint(0, 5, size=10)
    res1 = corr(x, y, alternative="less", seed=prng)
    res2 = corr(x, y, alternative="less")
    assert len(res1) == 3
    assert len(res2) == 3
    assert res1[0] == res2[0]
    np.testing.assert_almost_equal(res1[1], res2[1], decimal=1)
    print("finished test 2 in test_corr()")

    res1 = corr(x, y, alternative="two-sided", seed=prng)
    res2 = corr(x, y, alternative="greater")
    assert len(res1) == 3
    assert len(res2) == 3
    assert res1[0] == res2[0]
    np.testing.assert_almost_equal(res1[1], res2[1]*2, decimal=1)
    print("finished test 3 in test_corr()")


def test_spearman_corr():
    prng = SHA256(42)
    x = np.array([2, 4, 6, 8, 10])
    y = np.array([1, 3, 5, 6, 9])
    xorder = np.array([1, 2, 3, 4, 5])
    res1 = corr(xorder, xorder, seed=prng)
    print("finished test 1 in test_spearman_corr()")
    
    prng = SHA256(42)
    res2 = spearman_corr(x, y, seed=prng)
    assert res1[0] == res2[0]
    assert res1[1] == res2[1]
    np.testing.assert_array_equal(res1[2], res2[2])
    print("finished test 2 in test_spearman_corr()")


@pytest.mark.slow
def test_two_sample():
    prng = RandomState(42)

    # Normal-normal, different means examples
    x = prng.normal(1, size=20)
    y = prng.normal(4, size=20)
    res = two_sample(x, y, seed=42)
    expected = (1.0, -2.90532344604777)
    np.testing.assert_almost_equal(res, expected, 5)
    print("finished test 1 in test_two_sample()")

    res = two_sample(x, y, seed=42, plus1=False)
    np.testing.assert_almost_equal(res, expected)
    print("finished test 2 in test_two_sample()")

    # This one has keep_dist = True
    y = prng.normal(1.4, size=20)
    res = two_sample(x, y, seed=42)
    res2 = two_sample(x, y, seed=42, keep_dist=True)
    expected = (0.96975, -0.54460818906623765)
    np.testing.assert_almost_equal(res[0], expected[0], 2)
    assert res[1] == expected[1]
    np.testing.assert_almost_equal(res2[0], expected[0], 2)
    assert res2[1] == expected[1]
    print("finished test 3 in test_two_sample()")

    # Normal-normal, same means
    y = prng.normal(1, size=20)
    res = two_sample(x, y, seed=42)
    expected = (0.66505000000000003, -0.13990200413154097)
    np.testing.assert_almost_equal(res[0], expected[0], 2)
    assert res[1] == expected[1]
    print("finished test 4 in test_two_sample()")

    # Check the permutation distribution
    res = two_sample(x, y, seed=42, keep_dist=True)
    expected_pv = 0.66505000000000003
    expected_ts = -0.13990200413154097
    exp_dist_firstfive = [-0.1312181,  0.1289127, -0.3936627, -0.1439892,  0.7477683]
    np.testing.assert_almost_equal(res[0], expected_pv, 2)
    assert res[1] == expected_ts
    assert len(res[2]) == 100000
    np.testing.assert_almost_equal(res[2][:5], exp_dist_firstfive)
    print("finished test 5 in test_two_sample()")

    # Define a lambda function (K-S test)
    f = lambda u, v: np.max(
        [abs(sum(u <= val) / len(u) - sum(v <= val) / len(v))
         for val in np.concatenate([u, v])])
    res = two_sample(x, y, seed=42, stat=f, reps=100, plus1=False)
    expected = (0.62, 0.20000000000000007)
    assert res[0] == expected[0]
    assert res[1] == expected[1]
    print("finished test 6 in test_two_sample()")
    
    # check tail computations
    x = np.ones(10)
    y = np.ones(10)
    res = two_sample(x, y, reps=10**2, stat='mean', alternative="greater",
               keep_dist=False, seed=None, plus1=True)
    assert res[0] == 1
    assert res[1] == 0
    print("finished test 7 in test_two_sample()")
    
    res = two_sample(x, y, reps=10**2, stat='mean', alternative="less",
               keep_dist=False, seed=None, plus1=True)
    assert res[0] == 1
    assert res[1] == 0
    print("finished test 8 in test_two_sample()")

    res = two_sample(x, y, reps=10**2, stat='mean', alternative="two-sided",
               keep_dist=False, seed=None, plus1=True)
    assert res[0] == 1
    assert res[1] == 0
    print("finished test 9 in test_two_sample()")


def test_two_sample2():
    """
    Same as above but with a new dataset
    The expected p value is calculated by iterating through all possible permuation
    """

    brain_wts = np.array([ [689, 656, 668, 660, 679, 663, 664, 647, 694, 633, 653], \
              [657, 623, 652, 654, 658, 646, 600, 640, 605, 635, 642] ])
    p, obs, sim = two_permute(brain_wts[0], brain_wts[1], "two-sided", plus1=False)
    #print(p)
    diff_in_mean = lambda x, y: np.mean(x) - np.mean(y)
    res = two_sample(brain_wts[1], brain_wts[0], reps=10**5, stat=diff_in_mean, \
        alternative="two-sided", keep_dist=False, seed=12, plus1=False)
    #print(res)
    np.testing.assert_almost_equal(p, res[0], decimal=4)

    p, obs, sim = two_permute(brain_wts[0], brain_wts[1], "lower", plus1=False)
    #print(p)
    res = two_sample(brain_wts[1], brain_wts[0], reps=10**5, stat=diff_in_mean, \
        alternative="less", keep_dist=False, seed=12, plus1=False)
    #print(res)
    np.testing.assert_almost_equal(p, res[0], decimal=4)

    p, obs, sim = two_permute(brain_wts[0], brain_wts[1], "upper", plus1=False)
    #print(p)
    res = two_sample(brain_wts[1], brain_wts[0], reps=10**5, stat=diff_in_mean, \
        alternative="greater", keep_dist=False, seed=12, plus1=False)
    #print(res)
    np.testing.assert_almost_equal(p, res[0], decimal=4)

from itertools import combinations
def two_permute(treatment: list, control: list, alternative: str="two-sided", plus1: bool=True):
    ''' 
    Return two sample permutation test p-value

    Parameters
    ----------
    treatement : list, outcome of the treatment unit
    control    : list outcome of the controled unit, this is paired with the treatment
    alternative: str, alternative of the test
    
    Returns
    -------
    p : p value of the test
    observed : observed statistics
    simulated: simulated statistics
    '''
    # Find the difference
    assert alternative in ["lower", "upper", 'two-sided']
    treatment = np.array(treatment)
    control = np.array(control)
    observed = np.mean(treatment) - np.mean(control)
    
    combineddat = np.append(treatment, control)
    indecies = np.arange(len(combineddat))
    simulated = []
    
    
    # Iterates through all possible treatement assignment
    for comb in combinations(indecies, len(treatment)):
        treat = np.take(combineddat, comb)
        cont = np.delete(combineddat, comb)
        simulated += [np.mean(treat) - np.mean(cont)]
    
    simulated = np.array(simulated)
    # Adjust according to different alternative
    if alternative == 'two-sided':
        return (np.sum(abs(simulated) >= abs(observed))+plus1)/(len(simulated)+plus1), observed, simulated

    elif alternative == "lower":
        return (np.sum(simulated >= observed)+plus1)/(len(simulated)+plus1), observed, simulated

    else:
        return (np.sum(simulated <= observed)+plus1)/(len(simulated)+plus1), observed, simulated


@pytest.mark.slow
def test_two_sample_shift():
    prng = RandomState(42)

    # Normal-normal, different means examples
    x = prng.normal(1, size=20)
    y = prng.normal(4, size=20)
    f = lambda u: u - 3
    finv = lambda u: u + 3
    f_err = lambda u: 2 * u
    f_err_inv = lambda u: u / 2
    expected_ts = -2.9053234460477784

    # Test null with shift other than zero
    res = two_sample_shift(x, y, seed=42, shift=2, plus1=False)
    assert res[0] == 1
    assert res[1] == expected_ts
    print("finished test 1 in test_two_sample_shift()")

    res2 = two_sample_shift(x, y, seed=42, shift=2, keep_dist=True)
    np.testing.assert_almost_equal(res2[0], 1, 4)
    assert res2[1] == expected_ts
    np.testing.assert_almost_equal(res2[2][:3], np.array(
        [1.140174 , 2.1491466, 2.6169429]))
    print("finished test 2 in test_two_sample_shift()")

    res = two_sample_shift(x, y, seed=42, shift=2, alternative="less")
    np.testing.assert_almost_equal(res[0], 0, 3)
    assert res[1] == expected_ts
    print("finished test 3 in test_two_sample_shift()")

    # Test null with shift -3
    res = two_sample_shift(x, y, seed=42, shift=(f, finv))
    np.testing.assert_almost_equal(res[0], 0.377, 2)
    assert res[1] == expected_ts
    print("finished test 4 in test_two_sample_shift()")
    
    res = two_sample_shift(x, y, seed=42, shift=(f, finv), alternative="less")
    np.testing.assert_almost_equal(res[0], 0.622, 2)
    assert res[1] == expected_ts
    print("finished test 5 in test_two_sample_shift()")

    # Test null with multiplicative shift
    res = two_sample_shift(x, y, seed=42,
        shift=(f_err, f_err_inv), alternative="two-sided")
    np.testing.assert_almost_equal(res[0], 0, 3)
    assert res[1] == expected_ts
    print("finished test 6 in test_two_sample_shift()")

    # Define a lambda function
    f = lambda u, v: np.max(u) - np.max(v)
    res = two_sample(x, y, seed=42, stat=f, reps=100)
    expected = (1, -3.2730653690015465)
    assert res[0] == expected[0]
    assert res[1] == expected[1]
    print("finished test 7 in test_two_sample_shift()")


def test_two_sample_bad_shift():
    # Break it with a bad shift
    x = np.array(range(5))
    y = np.array(range(1, 6))
    shift = lambda u: u + 3
    pytest.raises(ValueError, two_sample_shift, x, y, seed=5, shift=shift)

"""
@pytest.mark.slow
def test_two_sample_conf_int():
    prng = RandomState(42)

    # Shift is -1
    x = np.array(range(5))
    y = np.array(range(1, 6))
    res = two_sample_conf_int(x, y, seed=prng)
    expected_ci = (-3.5, 1.0012461)
    np.testing.assert_almost_equal(res, expected_ci)
    print("finished test 1 in test_two_sample_conf_int()")

    res = two_sample_conf_int(x, y, seed=prng, alternative="upper")
    expected_ci = (-5, 1)
    np.testing.assert_almost_equal(res, expected_ci)
    print("finished test 2 in test_two_sample_conf_int()")
    
    res = two_sample_conf_int(x, y, seed=prng, alternative="lower")
    expected_ci = (-3, 5)
    np.testing.assert_almost_equal(res, expected_ci)
    print("finished test 3 in test_two_sample_conf_int()")

    # Specify shift with a function pair
    shift = (lambda u, d: u + d, lambda u, d: u - d)
    res = two_sample_conf_int(x, y, seed=5, shift=shift)
    np.testing.assert_almost_equal(res, (-3.5, 1))
    print("finished test 4 in test_two_sample_conf_int()")

    # Specify shift with a multiplicative pair
    shift = (lambda u, d: u * d, lambda u, d: u / d)
    res = two_sample_conf_int(x, y, seed=5, shift=shift)
    np.testing.assert_almost_equal(res, (-1, -1))
    print("finished test 5 in test_two_sample_conf_int()")
"""

def test_two_sample_conf_int_bad_shift():
    # Break it with a bad shift
    x = np.array(range(5))
    y = np.array(range(1, 6))
    shift = (lambda u, d: -d * u, lambda u, d: -u / d)
    pytest.raises(AssertionError, two_sample_conf_int, x, y, seed=5, shift=shift)


def test_one_sample():
    prng = RandomState(42)
    x = np.array(range(5))
    y = x - 1

    # case 1: one sample only
    res = one_sample(x, seed=42, reps=10**5, plus1=False)
    np.testing.assert_almost_equal(res[0], 2/32, decimal=2)
    assert res[1] == 2
    print("finished test 1 in test_one_sample()")

    res = one_sample(x, seed=42, reps=10**5, plus1=True)
    np.testing.assert_almost_equal(res[0], ((2/32)*(10**5) + 1)/((10**5) + 1), decimal = 2)
    assert res[1] == 2
    print("finished test 2 in test_one_sample()")

    # case 2: paired sample
    res = one_sample(x, y, seed=42, reps=10**5, keep_dist=True, plus1=False)
    np.testing.assert_almost_equal(res[0], 1/32, decimal=2)
    assert res[1] == 1
    assert min(res[2]) == -1
    assert max(res[2]) == 1
    print("finished test 3 in test_one_sample()")

    # case 3: break it - supply x and y, but not paired
    y = np.append(y, 10)
    pytest.raises(ValueError, one_sample, x, y)
    print("finished test 4 in test_one_sample()")

    # case 4: say keep_dist=True
    res = one_sample(x, seed=42, reps=10**5, keep_dist=True, plus1=False)
    np.testing.assert_almost_equal(res[0], 2/32, decimal=2)
    assert res[1] == 2
    assert min(res[2]) == -2
    assert max(res[2]) == 2
    assert np.median(res[2]) == 0
    print("finished test 5 in test_one_sample()")

    # case 5: use t as test statistic
    res = one_sample(x, y=None, seed=42, reps=10**5, stat="t", alternative="greater", plus1=False)
    np.testing.assert_almost_equal(res[0], 2/32, decimal=2)
    np.testing.assert_almost_equal(res[1], 2.82842712, decimal=2)
    print("finished test 6 in test_one_sample()")


def test_one_sample2():
    """
    Same as above but with a new dataset
    The expected p value is calculated by iterating through all possible permuation
    """

    brain_wts = np.array([ [689, 656, 668, 660, 679, 663, 664, 647, 694, 633, 653], \
              [657, 623, 652, 654, 658, 646, 600, 640, 605, 635, 642] ])
    p, obs, sim = one_paired_sample_permute(brain_wts[0], brain_wts[1], "two-sided", plus1=False)
    #print(p)
    sum_of_diff = lambda z: np.sum(z)
    res = one_sample(brain_wts[1], brain_wts[0], reps=10**5, stat=sum_of_diff, \
        alternative="two-sided", keep_dist=False, seed=42, plus1=False)
    #print(res)
    np.testing.assert_almost_equal(p, res[0], decimal=4)

    p, obs, sim = one_paired_sample_permute(brain_wts[0], brain_wts[1], "lower", plus1=False)
    #print(p)
    res = one_sample(brain_wts[1], brain_wts[0], reps=10**5, stat=sum_of_diff, \
        alternative="less", keep_dist=False, seed=42, plus1=False)
    #print(res)
    np.testing.assert_almost_equal(p, res[0], decimal=4)

    p, obs, sim = one_paired_sample_permute(brain_wts[0], brain_wts[1], "upper", plus1=False)
    #print(p)
    res = one_sample(brain_wts[1], brain_wts[0], reps=10**5, stat=sum_of_diff, \
        alternative="greater", keep_dist=False, seed=42, plus1=False)
    #print(res)
    np.testing.assert_almost_equal(p, res[0], decimal=4)
    

from itertools import product
import math
def one_paired_sample_permute(treatment: list, control: list, alternative: str="two-sided", plus1: bool=True):
    ''' 
    Return one sample permutation test p-value, assume the units are paired
    Test statistic is sum of differences

    Parameters
    ----------
    treatement : list, outcome of the treatment unit
    control    : list outcome of the controled unit, this is paired with the treatment
    alternative: str, alternative of the test
    
    Returns
    -------
    p : p value of the test
    observed : observed statistics
    simulated: simulated statistics
    '''
    # Find the difference
    assert alternative in ["lower", "upper", 'two-sided']
    diff = np.array(treatment) - np.array(control)
    observed = np.sum(diff)
    simulated = []
    
    # Iterates through all possible treatement assignment
    for signs in product([-1,1], repeat=len(diff)):
        # Simulated difference is signs * diff
        simulated += [sum(signs*diff)]
    
    simulated = np.array(simulated)
    # Adjust according to different alternative
    if alternative == 'two-sided':
        return (np.sum(abs(simulated) >= abs(observed))+plus1)/(len(simulated)+plus1), observed, simulated

    elif alternative == "lower":
        return (np.sum(simulated >= observed)+plus1)/(len(simulated)+plus1), observed, simulated

    else:
        return (np.sum(simulated <= observed)+plus1)/(len(simulated)+plus1), observed, simulated
