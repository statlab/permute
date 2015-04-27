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
    try:
        from .. import _test
        _test(dry_run=True)
    finally:
        sys.modules['nose'] = _tmp

def test_permute_tst():
    from .. import _test
    _test(dry_run=True)
    _test(doctest=True, dry_run=True)
    _test(doctest=True, verbose=True, dry_run=True)
