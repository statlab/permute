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
	expected_ci = (0.112977243, 1)
	np.testing.assert_almost_equal(res, expected_ci)

	res = hoeffding_conf_int(values, 10, 0, 1, cl=0.95)
	expected_ci = (0.070530591653262475, 0.92946940834673752)
	np.testing.assert_almost_equal(res, expected_ci)

	lb = np.array(list([0]*10))
	ub = np.array(list([1]*10))
	res = hoeffding_conf_int(values, 10, lb, up, cl=0.95, alternative="two-sided")
	expected_ci = (0.070530591653262475, 0.92946940834673752)
	np.testing.assert_almost_equal(res, expected_ci)

@raises(ValueError)
def test_hoeffding_conf_int_bad_bound():
    # Break it with a bad shift
    values = np.array(range(10))
    res = hoeffding_conf_int(values, 0, 9, 5, cl=0.95, alternative="two-sided")





		