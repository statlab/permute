from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from nose.plugins.attrib import attr
from nose.tools import assert_raises, raises

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_equal, assert_almost_equal, assert_array_equal
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
    assert_equal(len(res1), 3)
    assert_equal(len(res2), 3)
    assert_equal(res1[0], 1)
    assert_equal(res2[0], 1)
    assert_almost_equal(res1[1], res2[1], decimal=1)

    y = prng.randint(0, 5, size=10)
    res1 = corr(x, y, alternative="less", seed=prng)
    res2 = corr(x, y, alternative="less")
    assert_equal(len(res1), 3)
    assert_equal(len(res2), 3)
    assert_equal(res1[0], res2[0])
    assert_almost_equal(res1[1], res2[1], decimal=1)

    res1 = corr(x, y, alternative="two-sided", seed=prng)
    res2 = corr(x, y, alternative="greater")
    assert_equal(len(res1), 3)
    assert_equal(len(res2), 3)
    assert_equal(res1[0], res2[0])
    assert_almost_equal(res1[1], res2[1]*2, decimal=1)
    

def test_spearman_corr():
    prng = SHA256(42)
    x = np.array([2, 4, 6, 8, 10])
    y = np.array([1, 3, 5, 6, 9])
    xorder = np.array([1, 2, 3, 4, 5])
    res1 = corr(xorder, xorder, seed=prng)
    
    prng = SHA256(42)
    res2 = spearman_corr(x, y, seed=prng)
    assert_equal(res1[0], res2[0])
    assert_equal(res1[1], res2[1])
    assert_array_equal(res1[2], res2[2])


@attr('slow')
def test_two_sample():
    prng = RandomState(42)

    # Normal-normal, different means examples
    x = prng.normal(1, size=20)
    y = prng.normal(4, size=20)
    res = two_sample(x, y, seed=42)
    expected = (1.0, -2.90532344604777)
    assert_almost_equal(res, expected, 5)
    res = two_sample(x, y, seed=42, plus1=False)
    assert_almost_equal(res, expected)

    # This one has keep_dist = True
    y = prng.normal(1.4, size=20)
    res = two_sample(x, y, seed=42)
    res2 = two_sample(x, y, seed=42, keep_dist=True)
    expected = (0.96975, -0.54460818906623765)
    assert_almost_equal(res[0], expected[0], 2)
    assert_equal(res[1], expected[1])
    assert_almost_equal(res2[0], expected[0], 2)
    assert_equal(res2[1], expected[1])

    # Normal-normal, same means
    y = prng.normal(1, size=20)
    res = two_sample(x, y, seed=42)
    expected = (0.66505000000000003, -0.13990200413154097)
    assert_almost_equal(res[0], expected[0], 2)
    assert_equal(res[1], expected[1])

    # Check the permutation distribution
    res = two_sample(x, y, seed=42, keep_dist=True)
    expected_pv = 0.66505000000000003
    expected_ts = -0.13990200413154097
    exp_dist_firstfive = [-0.1312181,  0.1289127, -0.3936627, -0.1439892,  0.7477683]
    assert_almost_equal(res[0], expected_pv, 2)
    assert_equal(res[1], expected_ts)
    assert_equal(len(res[2]), 100000)
    assert_almost_equal(res[2][:5], exp_dist_firstfive)

    # Define a lambda function (K-S test)
    f = lambda u, v: np.max(
        [abs(sum(u <= val) / len(u) - sum(v <= val) / len(v))
         for val in np.concatenate([u, v])])
    res = two_sample(x, y, seed=42, stat=f, reps=100, plus1=False)
    expected = (0.62, 0.20000000000000007)
    assert_equal(res[0], expected[0])
    assert_equal(res[1], expected[1])
    
    # check tail computations
    x = np.ones(10)
    y = np.ones(10)
    res = two_sample(x, y, reps=10**2, stat='mean', alternative="greater",
               keep_dist=False, seed=None, plus1=True)
    assert_equal(res[0], 1)
    assert_equal(res[1], 0)
    res = two_sample(x, y, reps=10**2, stat='mean', alternative="less",
               keep_dist=False, seed=None, plus1=True)
    assert_equal(res[0], 1)
    assert_equal(res[1], 0)
    res = two_sample(x, y, reps=10**2, stat='mean', alternative="two-sided",
               keep_dist=False, seed=None, plus1=True)
    assert_equal(res[0], 1)
    assert_equal(res[1], 0)
    


@attr('slow')
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
    assert_equal(res[0], 1)
    assert_equal(res[1], expected_ts)
    res2 = two_sample_shift(x, y, seed=42, shift=2, keep_dist=True)
    assert_almost_equal(res2[0], 1, 4)
    assert_equal(res2[1], expected_ts)
    assert_almost_equal(res2[2][:3], np.array(
        [1.140174 , 2.1491466, 2.6169429]))
    res = two_sample_shift(x, y, seed=42, shift=2, alternative="less")
    assert_almost_equal(res[0], 0, 3)
    assert_equal(res[1], expected_ts)

    # Test null with shift -3
    res = two_sample_shift(x, y, seed=42, shift=(f, finv))
    assert_almost_equal(res[0], 0.377, 2)
    assert_equal(res[1], expected_ts)
    res = two_sample_shift(x, y, seed=42, shift=(f, finv), alternative="less")
    assert_almost_equal(res[0], 0.622, 2)
    assert_equal(res[1], expected_ts)

    # Test null with multiplicative shift
    res = two_sample_shift(x, y, seed=42,
        shift=(f_err, f_err_inv), alternative="two-sided")
    assert_almost_equal(res[0], 0, 3)
    assert_equal(res[1], expected_ts)

    # Define a lambda function
    f = lambda u, v: np.max(u) - np.max(v)
    res = two_sample(x, y, seed=42, stat=f, reps=100)
    expected = (1, -3.2730653690015465)
    assert_equal(res[0], expected[0])
    assert_equal(res[1], expected[1])


@raises(ValueError)
def test_two_sample_bad_shift():
    # Break it with a bad shift
    x = np.array(range(5))
    y = np.array(range(1, 6))
    shift = lambda u: u + 3
    two_sample_shift(x, y, seed=5, shift=shift)

@attr('slow')
def test_two_sample_conf_int():
    # we want to find the largest number such that the p-value is greater than or equal to alpha

    seed = 42
    prng = SHA256(seed)
    
    #shift is -2
    x = np.array([1,2])
    y = np.array([3,4]) 
    res = two_sample_conf_int(x, y, cl=0.9, seed = prng)
    expected_ci = (-3, -1)
    assert_almost_equal(res, expected_ci)
    res = two_sample_conf_int(x, y, cl=0.65, seed = prng)
    expected_ci = (-3, 3)
    assert_almost_equal(res, expected_ci)
    #res = two_sample_conf_int(x, y, cl=0.9, seed = prng, stat = 't')
    #expected_ci = (-3, -1)
    #assert_almost_equal(res, expected_ci)
    
    
    #shift is 2
    x = np.array([3,4,5])
    y = np.array([1,2,3])
    res = two_sample_conf_int(x, y, cl=0.91, seed = prng)
    expected_ci = (0, 4)
    assert_almost_equal(res, expected_ci)
    
    
    # Shift is -1
    x = np.array(range(5))
    y = np.array(range(1, 6))
    res = two_sample_conf_int(x, y, seed=prng)
    expected_ci = (-3, 1)
    assert_almost_equal(res, expected_ci)
    res = two_sample_conf_int(x, y, seed=prng, alternative="upper")
    expected_ci = (-5, 1)
    assert_almost_equal(res, expected_ci)
    res = two_sample_conf_int(x, y, seed=prng, alternative="lower")
    expected_ci = (-3, 5)
    assert_almost_equal(res, expected_ci)
    res = two_sample_conf_int(x, y, cl=0.7, seed = prng)
    expected_ci = (-2, 0)
    assert_almost_equal(res, expected_ci)

    # Specify shift with a function pair
    #shift = (lambda u, d: u + d, lambda u, d: u - d)
    #res = two_sample_conf_int(x, y, seed=5, shift=shift)
    #assert_almost_equal(res, (-3.5, 1))

    # Specify shift with a multiplicative pair
    #shift = (lambda u, d: u * d, lambda u, d: u / d)
    #res = two_sample_conf_int(x, y, seed=5, shift=shift)
    #assert_almost_equal(res, (-1, -1))


@raises(AssertionError)
def test_two_sample_conf_int_bad_shift():
    # Break it with a bad shift
    x = np.array(range(5))
    y = np.array(range(1, 6))
    shift = (lambda u, d: -d * u, lambda u, d: -u / d)
    two_sample_conf_int(x, y, seed=5, shift=shift)


@raises(AssertionError)
def test_two_sample_conf_int_bad_shift():
    # Break it with a bad shift
    seed = 42
    x = np.array(range(5))
    y = np.array(range(1, 6))
    shift = (lambda u, d: -d * u, lambda u, d: -u / d)
    two_sample_conf_int(x, y, seed= seed, shift=shift)


def test_one_sample():
    seed = 42
    reps=1000000
    
    x = np.array(range(5))
    y = x - 1

    # case 1: one sample only
    res = one_sample(x, seed=seed, reps=reps, plus1=False) 
    assert_almost_equal(res[0], 1/(2**4), decimal = 4) #ACTUAL: 0.06247, DESIRED:0.0625
    assert_equal(res[1], 2)
    res = one_sample(x, seed=seed, reps=reps, plus1=True)
    assert_almost_equal(res[0], (1/(2**4)*reps)/(reps+1)+(1/(reps+1)), decimal = 4)#ACTUAL:0.06247093752906247, DESIRED: 0.06250093749906249
    assert_equal(res[1], 2)

    # case 2: paired sample
    res = one_sample(x, y, seed=seed, reps=reps, plus1=False)
    assert_almost_equal(res[0],1/(2**5), decimal = 4) #ACTUAL:0.031225, DESIRED: 0.03125
    assert_equal(res[1], 1)

    # case 3: break it - supply x and y, but not paired
    y = np.append(y, 10)
    assert_raises(ValueError, one_sample, x, y)

    # case 4: say keep_dist=True
    res = one_sample(x, seed=seed, reps=reps, keep_dist=True, plus1=False)
    assert_almost_equal(res[0], 1/(2**4), decimal = 4)
    assert_equal(res[1], 2)
    assert_equal(min(res[2]), -2)
    assert_equal(max(res[2]), 2)
    assert_equal(np.median(res[2]), 0)

    
    #case 5: use t as test statistic
    y = np.array(x)- np.array(range(-1,4)) #y=1,1,1,1,1, z= -1,0,1,2,3
    res = one_sample(x, y, seed = seed, reps=reps, stat="t", alternative="less", plus1=False)
    assert_almost_equal(res[0], 1-1/16, decimal = 4) 
    assert_almost_equal(res[1], 1.414213562373095)
    
    #case 6: one sample two sided
    x = np.array([1,1,1,1,1,0])
    res = one_sample(x, seed=seed, reps=reps, plus1=False,alternative="two-sided") 
    assert_almost_equal(res[0],2/(2**5), decimal = 3) #ACTUAL:0.062292, DESIRED: 0.0625
    assert_equal(res[1], 5/6)
    
    #case 7: one sample less
    x = np.array([-1,2,3,4,5,0])
    res = one_sample(x, seed=seed, reps=reps, plus1=False,alternative="less") 
    assert_almost_equal(res[0],1-1/(2**5), decimal = 4)
    assert_equal(res[1], 13/6)
