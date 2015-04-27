from __future__ import division, print_function, absolute_import

import sys

import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal)

from nose.tools import raises

@raises(ImportError)
def test_nose_import_error():
    _tmp = sys.modules['nose']
    sys.modules['nose'] = None
    from .. import _test
    _test(dry_run=True)
    sys.modules['nose'] = _tmp
