from ..irr import compute_ts, simulate_ts_dist
#from nose.tools import assert_almost_equal

import numpy as np
from numpy.testing import (assert_equal,
                           assert_array_equal,
                           assert_almost_equal)


R = 10
Ns = 35

np.random.seed(42)
res =  np.random.binomial(1, .5, (R, Ns))

def test_irr():
    rho_s = compute_ts(res)
    assert_almost_equal(rho_s, 0.51936507)
    #res = spt(group, condition, response, iterations=1000) 
    #res1 = spt(group, condition, response, iterations=1000) 
    #assert_less(res[1], 0.01)
    #assert_almost_equal(res[3], res1[3])

def test_simulate_ts_dist():
    expected_res1 = {'dist': None,
                    'geq': 615,
                    'obs_ts': 0.51936507936507936,
                    'iter': 10000}
    res1 = simulate_ts_dist(res)
    assert_equal(res1, expected_res1)
    


np.random.seed(55)
freq = np.random.choice([0.2, 0.8], Ns)
res2 = np.zeros((R, Ns))

for i in range(len(freq)):
    res2[:,i] = np.random.binomial(1, freq[i], R)

def test_irr_concordance():
    rho_s2 = compute_ts(res2)
    assert_almost_equal(rho_s2, 0.67619047619047623)
    
def test_simulate_ts_dist_concordance():
    expected_res_conc = {'dist': None,
                    'geq': 0,
                    'obs_ts': 0.67619047619047623,
                    'iter': 10000}
    res_conc = simulate_ts_dist(res2)
    assert_equal(res_conc, expected_res_conc)