"""Permutation tests and confidence intervals in Python

``permute`` provides permutation tests and confidence intervals for a variety
of nonparametric testing and estimation problems, for a variety of
randomization designs.

"""

import os.path as _osp
import imp as _imp
import functools as _functools
import warnings as _warnings

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')

try:
    from .version import version as __version__
except ImportError:
    __version__ = "unbuilt-dev"
else:
    del version


try:
    _imp.find_module('nose')
except ImportError:
    def _test(verbose=False):
        """This would run all unit tests, but nose couldn't be
        imported so the test suite can not run.
        """
        raise ImportError("Could not load nose. Unit tests not available.")

    def _doctest(verbose=False):
        """This would run all doc tests, but nose couldn't be
        imported so the test suite can not run.
        """
        raise ImportError("Could not load nose. Doctests not available.")
else:
    def _test(doctest=False, verbose=False):
        """Run all unit tests."""
        import nose
        args = ['', pkg_dir, '--exe', '--ignore-files=^_test']
        if verbose:
            args.extend(['-v', '-s'])
        if doctest:
            args.extend(['--with-doctest', '--ignore-files=^\.',
                         '--ignore-files=^setup\.py$$', '--ignore-files=test'])
            # Make sure warnings do not break the doc tests
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                success = nose.run('permute', argv=args)
        else:
            success = nose.run('permute', argv=args)
        # Return sys.exit code
        if success:
            return 0
        else:
            return 1


# do not use `test` as function name as this leads to a recursion problem with
# the nose test suite
test = _test
test_verbose = _functools.partial(test, verbose=True)
test_verbose.__doc__ = test.__doc__
doctest = _functools.partial(test, doctest=True)
doctest.__doc__ = doctest.__doc__
doctest_verbose = _functools.partial(test, doctest=True, verbose=True)
doctest_verbose.__doc__ = doctest.__doc__
