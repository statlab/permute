from __future__ import division, print_function, absolute_import

import contextlib
import sys

from numpy.testing import assert_equal

from nose.tools import raises


# def setup(module):
#     module._mods = sys.modules.copy()
#     module._tmp = sys.modules['nose']
#     sys.modules['nose'] = None
#     sys.modules['permute'] = reload(sys.modules['permute'])
#
# def teardown(module):
#     to_del = [ m for m in sys.modules.keys() if m not in
#                module._mods ]
#     for mod in to_del:
#         del sys.modules[mod]
#     sys.modules['nose'] = _tmp
#     sys.modules['nose'] = reload(sys.modules['nose'])
#     sys.modules['permute'] = reload(sys.modules['permute'])
#     sys.modules.update(module._mods)


class DummyFile(object):

    def write(self, x):
        pass


# @raises(ImportError)
# def test_del_version():
#     _tmp = sys.modules['nose']
#     sys.modules['nose'] = None
#     try:
#         from .. import __version__
#     finally:
#         sys.modules['nose'] = _tmp


def test_version():
    from .. import __version__ as v
    try:
        from ..version import version as _v
    except ImportError:
        assert_equal(v, "unbuilt-dev")
    else:
        assert_equal(v, _v)


@raises(ImportError)
def test_nose_import_error1():
    _tmp = sys.modules['nose']
    sys.modules['nose'] = None
    try:
        from .. import test
        test()
    finally:
        sys.modules['nose'] = _tmp


# @raises(ImportError)
# def test_nose_import_error2():
#     _tmp = sys.modules['nose']
#     sys.modules['nose'] = None
#     try:
#         from .. import doctest
#         doctest()
#     finally:
#         sys.modules['nose'] = _tmp


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
