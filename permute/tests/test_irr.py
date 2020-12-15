import pytest

import numpy as np

from ..irr import (compute_ts,
                   simulate_ts_dist,
                   simulate_npc_dist)

from ..data import nsgk

R = 10
Ns = 35

from numpy.random import RandomState
RNG = RandomState(42)
res = RNG.binomial(1, .5, (R, Ns))


def test_irr():
    rho_s = compute_ts(res)
    np.testing.assert_almost_equal(rho_s, 0.51936507)


def test_simulate_ts_dist():
    expected_res1 = {'dist': None,
                     'geq': 591,
                     'obs_ts': 0.51936507936507936,
                     'pvalue': 0.0591,
                     'num_perm': 10000}
    res1 = simulate_ts_dist(res, seed=42, plus1=False)
    np.testing.assert_equal(res1, expected_res1)
    expected_res2 = {'geq': 9507,
                     'obs_ts': 0.46285714285714286,
                     'num_perm': 10000}
    res2 = simulate_ts_dist(res[:5], seed=42, keep_dist=True)
    assert res2['geq'] == expected_res2['geq']
    assert res2['obs_ts'] == expected_res2['obs_ts']
    assert res2['num_perm'] == expected_res2['num_perm']
    assert res2['dist'].shape == (10000,)


def test_with_naomi_data():
    """ Test irr functionality using Naomi data."""
    x = nsgk()
    t = x[1]
    y = t[0]
    res = simulate_ts_dist(y, num_perm=10, keep_dist=True, seed=42, plus1=False)
    expected_res = {'dist': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
                    'geq': 10,
                    'num_perm': 10,
                    'pvalue': 1,
                    'obs_ts': 1.0}
    np.testing.assert_equal(res, expected_res)


freq = RNG.choice([0.2, 0.8], Ns)
res2 = np.zeros((R, Ns))

for i in range(len(freq)):
    res2[:, i] = RNG.binomial(1, freq[i], R)


def test_irr_concordance():
    rho_s2 = compute_ts(res2)
    np.testing.assert_almost_equal(rho_s2, 0.70476190476190481)


def test_simulate_ts_dist_concordance():
    expected_res_conc = {'dist': None,
                         'geq': 0,
                         'obs_ts': 0.70476190476190481,
                         'pvalue': 1/10001,
                         'num_perm': 10000}
    res_conc = simulate_ts_dist(res2, seed=42)
    np.testing.assert_equal(res_conc, expected_res_conc)


res1 = simulate_ts_dist(res, keep_dist=True, seed=42)
res_conc = simulate_ts_dist(res2, keep_dist=True, seed=42)
true_pvalue = np.array(
    [res1['geq'] / res1['num_perm'], res_conc['geq'] / res_conc['num_perm']])
rho_perm = np.transpose(np.vstack((res1['dist'], res_conc['dist'])))


def test_simulate_npc_dist():
    expected_npc_res = {'num_perm': 10000,
                        'obs_npc': -0.00998,
                        'pvalue': 0.0016}
    obs_npc_res = simulate_npc_dist(
        rho_perm, size=np.array([Ns, Ns]), pvalues=true_pvalue)
    assert obs_npc_res['num_perm'] == expected_npc_res['num_perm']
    np.testing.assert_almost_equal(obs_npc_res['obs_npc'], expected_npc_res['obs_npc'], 3)
    np.testing.assert_almost_equal(obs_npc_res['pvalue'], expected_npc_res['pvalue'], 3)


def test_simulate_npc_error():
    pytest.raises(ValueError, simulate_npc_dist, rho_perm, size=np.array([Ns, Ns]))


def test_simulate_npc_perfect():
    mat1 = np.tile(np.array([1, 0, 1, 0, 0]), (5, 1))
    mat2 = np.tile(np.array([0, 1, 0]), (5, 1))
    videos = [mat1, mat2]
    time_stamps = np.array([5, 3])
    d = []          # list of the permutation distributions for each video
    tst = []        # list of test statistics for each video
    pval = []
    for j in range(len(videos)):  # loop over videos
        res = simulate_ts_dist(videos[j], keep_dist=True, seed=5)
        d.append(res['dist'])
        tst.append(res['obs_ts'])
        pval.append(res['pvalue'])
    perm_distr = np.asarray(d).transpose()
    overall1 = simulate_npc_dist(
        perm_distr, size=time_stamps, pvalues=np.array(pval), plus1=False)
    overall2 = simulate_npc_dist(
        perm_distr, size=time_stamps, obs_ts=tst)
    expected_overall = {'num_perm': 10000,
                        'obs_npc': -0.007709695302872763,
                        'pvalue': 0.0}
    np.testing.assert_almost_equal(overall1['obs_npc'], expected_overall['obs_npc'], 3)
    np.testing.assert_almost_equal(overall2['obs_npc'], expected_overall['obs_npc'], 3)
