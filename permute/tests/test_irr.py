from ..irr import compute_ts
from nose.tools import assert_almost_equal

import numpy as np
#from numpy.testing import (assert_array_equal,
#                           assert_almost_equal)


def test_irr():
    R = 10
    Ns = 35

    np.random.seed(42)
    res =  np.random.binomial(1, .5, (R, Ns))
    rho_s = compute_ts(res)
    assert_almost_equal(rho_s, 0.51936507)
    #res = spt(group, condition, response, iterations=1000) 
    #res1 = spt(group, condition, response, iterations=1000) 
    #assert_less(res[1], 0.01)
    #assert_almost_equal(res[3], res1[3])
