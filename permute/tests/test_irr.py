from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal)

from ..irr import (compute_ts,
                   simulate_ts_dist,
                   compute_inverseweight_npc,
                   simulate_npc_dist)


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
                     'iter': 10000}
    res1 = simulate_ts_dist(res, seed=42)
    assert_equal(res1, expected_res1)


freq = RNG.choice([0.2, 0.8], Ns)
res2 = np.zeros((R, Ns))

for i in range(len(freq)):
    res2[:, i] = RNG.binomial(1, freq[i], R)


def test_irr_concordance():
    rho_s2 = compute_ts(res2)
    assert_almost_equal(rho_s2, 0.70476190476190481)


#@np.testing.decorators.skipif(True)
def test_simulate_ts_dist_concordance():
    expected_res_conc = {'dist': None,
                         'geq': 0,
                         'obs_ts': 0.70476190476190481,
                         'iter': 10000}
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
    [res1['geq'] / res1['iter'], res_conc['geq'] / res_conc['iter']])
rho_perm = np.transpose(np.vstack((res1['dist'], res_conc['dist'])))


def test_simulate_npc_dist():
    expected_npc_res = {'dist': None,
                        'iter': 10000,
                        'leq': 5,
                        'obs_npc':  0.010547525099011886}
    obs_npc_res = simulate_npc_dist(
        rho_perm, size=np.array([Ns, Ns]), pvalues=true_pvalue)
    assert_equal(obs_npc_res, expected_npc_res)
