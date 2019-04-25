from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from nose.plugins.attrib import attr
from nose.tools import assert_raises, raises

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_equal, assert_almost_equal, assert_array_less
from scipy.stats import hypergeom, binom
from cryptorandom.cryptorandom import SHA256

from ..ksample import (k_sample, one_way_anova)
import permute.data as data

def test_worms_ksample():
    worms = data.worms()
    res = k_sample(worms.x, worms.y, stat='one-way anova')
    assert_array_less(0.005, res[0])
    assert_array_less(res[0], 0.02)


def test_one_way_anova():
    group = np.ones(5)
    x = np.array(range(5))
    xbar = np.mean(x)
    assert_equal(one_way_anova(x, group, xbar), 0)
    
    group = np.array([1]*3 + [2]*2)
    expected = 3*1**2 + 2*1.5**2
    assert_equal(one_way_anova(x, group, xbar), expected)