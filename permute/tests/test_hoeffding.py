from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from nose.plugins.attrib import attr
from nose.tools import assert_raises, raises

import numpy as np
from numpy.random import RandomState


from ..hoeffding import hoeffding_conf_int

def test_hoeffding():
	values = np.array([0.5]*10)
	res = hoeffding_conf_int(values, 10, 0, 1, cl=0.95, alternative="two-sided")
	expected_ci = (0.070530591653262475, 0.92946940834673752)
	np.testing.assert_almost_equal(res, expected_ci)

	res = hoeffding_conf_int(values, 10, 0, 1, cl=0.95)
	expected_ci = (0.11297724397950515, 1)
	np.testing.assert_almost_equal(res, expected_ci)

	lb = np.array(list([0]*10))
	ub = np.array(list([1]*10))
	res = hoeffding_conf_int(values, 10, lb, ub, cl=0.95, alternative="two-sided")
	expected_ci = (0.070530591653262475, 0.92946940834673752)
	np.testing.assert_almost_equal(res, expected_ci)


def test_hoeffding_conf_int_bad_bound():
    # Break it with incorrect upper and lower bounds
    x = np.array(range(10))
    assert_raises(ValueError, hoeffding_conf_int, x, 10, 5, 6)

    # Invalid upper and lower bounds
    upper_bound = 1
    lower_bound = np.array([10, 10, 10])
    assert_raises(ValueError, hoeffding_conf_int, x, 10, lower_bound, upper_bound)

    # Bad bound input length
    upper_bound = np.array([10, 10])
    lower_bound = np.array([10, 10])
    assert_raises(ValueError, hoeffding_conf_int, x, 10, lower_bound, upper_bound)

    # bounds are not value 
    upper_bound = np.array([10]*10)
    lower_bound = np.array([20]*10)
    assert_raises(ValueError, hoeffding_conf_int, x, 10, lower_bound, upper_bound)

    # x not in range of bounds
    x = np.array([5]*10)
    upper_bound = np.array([3]*10)
    lower_bound = np.array([2]*10)
    assert_raises(ValueError, hoeffding_conf_int, x, 10, lower_bound, upper_bound)

    






		