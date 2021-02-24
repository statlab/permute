import contextlib
import sys

import pytest


class DummyFile:

    def write(self, x):
        pass



def test_pytest_import_error1():
    _tmp = sys.modules['pytest']
    sys.modules['pytest'] = None
    try:
        from .. import test
        pytest.raises(ImportError, test)
    finally:
        sys.modules['pytest'] = _tmp


@contextlib.contextmanager
def test_permute_tst():
    from .. import test
    save_stderr = sys.stderr
    sys.stderr = DummyFile()
    test(dry_run=True)
    test(doctest=True, dry_run=True)
    test(run_all=False, dry_run=True)
    test(doctest=True, verbose=True, dry_run=True)
    sys.stderr = save_stderr
