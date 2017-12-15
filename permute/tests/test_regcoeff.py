from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from nose.plugins.attrib import attr
from nose.tools import assert_raises, raises

import numpy as np
from numpy.random import RandomState

from ..regcoeff import reg_coeff

def test_regcoeff():
	values = np.array([0.5]*10)
	x = np.array(range(10))
	y = np.array(range(10))
	res = reg_coeff(x, y, cl = 0.95, seed=42)
	expected = 0
	np.testing.assert_almost_equal(res, expected)

	y.append(10)
	assert_raises(ValueError, reg_coeff, x, y)