from __future__ import division, print_function, absolute_import

import contextlib
import sys

import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal)

from nose.tools import raises

class DummyFile(object):
    def write(self, x): pass

@raises(ImportError)
def test_nose_import_error():
    _tmp = sys.modules['nose']
    sys.modules['nose'] = None
    try:
        from .. import _test
        _test(dry_run=True)
    finally:
        sys.modules['nose'] = _tmp

@contextlib.contextmanager
def test_permute_tst():
    from .. import _test
    save_stderr = sys.stderr
    sys.stderr = DummyFile()
    _test(dry_run=True)
    _test(doctest=True, dry_run=True)
    _test(run_all=False, dry_run=True)
    _test(doctest=True, verbose=True, dry_run=True)
    sys.stderr = save_stderr
