from __future__ import division, print_function, absolute_import

from nose.plugins.attrib import attr
from nose.tools import raises

import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal)

from ..irr import (compute_ts,
                   simulate_ts_dist,
                   compute_inverseweight_npc,
                   simulate_npc_dist)

from ..data import nsgk

R = 10
Ns = 35

from numpy.random import RandomState
RNG = RandomState(42)
res = RNG.binomial(1, .5, (R, Ns))


def test_irr():
    rho_s = compute_ts(res)
    assert_almost_equal(rho_s, 0.51936507)
    #res = spt(group, condition, response, iterations=1000)
    #res1 = spt(group, condition, response, iterations=1000)
    #assert_less(res[1], 0.01)
    #assert_almost_equal(res[3], res1[3])


def test_simulate_ts_dist():
    expected_res1 = {'dist': None,
                     'geq': 624,
                     'obs_ts': 0.51936507936507936,
                     'pvalue': 0.0624,
                     'num_perm': 10000}
    res1 = simulate_ts_dist(res, seed=42)
    assert_equal(res1, expected_res1)
    expected_res2 = {'geq': 9457,
                     'obs_ts': 0.46285714285714286,
                     'num_perm': 10000}
    res2 = simulate_ts_dist(res[:5], seed=42, keep_dist=True)
    assert_equal(res2['geq'], expected_res2['geq'])
    assert_equal(res2['obs_ts'], expected_res2['obs_ts'])
    assert_equal(res2['num_perm'], expected_res2['num_perm'])
    assert_equal(res2['dist'].shape, (10000,))


@attr('slow')
def test_with_naomi_data():
    """ Test irr functionality using Naomi data."""
    x = nsgk()
    t = x[1]
    y = t[0]
    res = simulate_ts_dist(y, num_perm=10, keep_dist=True, seed=42)
    expected_res = {'dist': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
                    'geq': 10,
                    'num_perm': 10,
                    'pvalue': 1,
                    'obs_ts': 1.0}
    assert_equal(res, expected_res)

    # FIXME: this is how the analysis will be run. I just put it in the test
    # file temporarily
    time_stamps = np.array([36, 32, 35, 37, 31, 35, 40, 32])
    category = []
    # for i in range(len(nsgk)):
    for i in range(20):  # loop over categories
        d = []          # list of the permutation distributions for each video
        tst = []        # list of test statistics for each video
        for j in range(len(x[i])):  # loop over videos
            res = simulate_ts_dist(x[i][j], keep_dist=True)
            d.append(res['dist'])
            tst.append(res['obs_ts'])
        perm_distr = np.asarray(d).transpose()
        category.append(
            simulate_npc_dist(perm_distr, size=time_stamps, obs_ts=tst, keep_dist=False))
    category_pvalues = []
    for i in range(len(category)):
        category_pvalues.append(category[i]['pvalue'])

freq = RNG.choice([0.2, 0.8], Ns)
res2 = np.zeros((R, Ns))

for i in range(len(freq)):
    res2[:, i] = RNG.binomial(1, freq[i], R)


def test_irr_concordance():
    rho_s2 = compute_ts(res2)
    assert_almost_equal(rho_s2, 0.70476190476190481)


def test_simulate_ts_dist_concordance():
    expected_res_conc = {'dist': None,
                         'geq': 0,
                         'obs_ts': 0.70476190476190481,
                         'pvalue': 0.0,
                         'num_perm': 10000}
    res_conc = simulate_ts_dist(res2, seed=42)
    assert_equal(res_conc, expected_res_conc)


pval = np.array([0.5, 0.25, 0.75])
size = np.array([2, 4, 6])


def test_compute_inverseweight_npc():
    expected_npc = 0.7847396
    res_npc = compute_inverseweight_npc(pval, size)
    assert_almost_equal(expected_npc, res_npc)


res1 = simulate_ts_dist(res, keep_dist=True, seed=42)
res_conc = simulate_ts_dist(res2, keep_dist=True, seed=42)
true_pvalue = np.array(
    [res1['geq'] / res1['num_perm'], res_conc['geq'] / res_conc['num_perm']])
rho_perm = np.transpose(np.vstack((res1['dist'], res_conc['dist'])))


def test_simulate_npc_dist():
    expected_npc_res = {'dist': None,
                        'num_perm': 10000,
                        'leq': 5,
                        'obs_npc': 0.010547525099011886,
                        'pvalue': 0.0005}
    obs_npc_res = simulate_npc_dist(
        rho_perm, size=np.array([Ns, Ns]), pvalues=true_pvalue)
    assert_equal(obs_npc_res, expected_npc_res)
    expected_npc_res1 = {'num_perm': 10000,
                         'leq': 5,
                         'obs_npc':  0.010547525099011886}
    obs_npc_res1 = simulate_npc_dist(
        rho_perm, size=np.array([Ns, Ns]), pvalues=true_pvalue, keep_dist=True)
    assert_equal(obs_npc_res1['num_perm'], expected_npc_res1['num_perm'])
    assert_equal(obs_npc_res1['leq'], expected_npc_res1['leq'])
    assert_equal(obs_npc_res1['obs_npc'], expected_npc_res1['obs_npc'])
    assert_equal(len(obs_npc_res1), 5)
    assert_almost_equal(
        obs_npc_res1['dist'][:2], np.array([0.5820746,  0.1648727]))


@raises(ValueError)
def test_simulate_npc_error():
    simulate_npc_dist(rho_perm, size=np.array([Ns, Ns]))


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
        perm_distr, size=time_stamps, pvalues=pval, keep_dist=False)
    overall2 = simulate_npc_dist(
        perm_distr, size=time_stamps, obs_ts=tst, keep_dist=False)
    expected_overall = {'dist': None,
                        'leq': 83,
                        'num_perm': 10000,
                        'obs_npc': 0.0076080098859340932,
                        'pvalue': 0.0083000000000000001}
    assert_equal(overall1, expected_overall)
    assert_equal(overall2, expected_overall)

    overall = simulate_npc_dist(
        perm_distr, size=time_stamps, obs_ts=tst, keep_dist=True)
    exp_firstfive = np.array(
        [0.02223304,  0.67969567,  1.13757326,  0.67969567,  0.67969567])
    assert_almost_equal(overall['dist'].mean(), 0.79636768325946183)
    assert_almost_equal(overall['dist'][:5], exp_firstfive)
