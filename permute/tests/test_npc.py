from __future__ import division, print_function, absolute_import

from nose.plugins.attrib import attr
from nose.tools import assert_raises, raises

import numpy as np
from scipy.stats import norm

from ..npc import (fisher,
                   liptak,
                   tippett)

def test_fisher():
    pvalues = np.linspace(0.05, 0.9, num=5)
    np.testing.assert_almost_equal(fisher(pvalues), 11.11546, 5)
    np.testing.assert_equal(fisher(1), -0.0)
    np.testing.assert_array_less(fisher(10), 0)
    
    
def test_liptak():
    pvalues = np.linspace(0.05, 0.9, num=5)
    np.testing.assert_almost_equal(liptak(pvalues), 0.5728894, 5)
    np.testing.assert_equal(liptak(1), norm.ppf(0))
    np.testing.assert_equal(liptak(10), np.nan)
    

def test_tippett():
    pvalues = np.linspace(0.05, 0.9, num=5)
    np.testing.assert_almost_equal(tippett(pvalues), 0.95, 5)
    np.testing.assert_equal(tippett(1), 0)
    np.testing.assert_equal(tippett(10), -9)
    