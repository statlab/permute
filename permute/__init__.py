"""
``permute`` provides permutation tests and confidence intervals for a variety
of nonparametric testing and estimation problems, for a variety of
randomization designs.

* Stratified and unstratified tests
* Test statistics in each stratum
* Methods of combining tests across strata
* Nonparametric combinations of tests

Problems/Methods:
-----------------

1. The 2-sample problem
2. The *n*-sample problem
3. Tests for the slope in linear regression
4. Tests for quantiles
5. Tests of independence and association: runs tests, permutation association
6. Tests of exchangeability
7. Tests of symmetry: reflection, spherical
8. Permutation ANOVA
9. Goodness of fit tests


Confidence sets
---------------

1. Constant shifts
2. Proportional shifts
3. Monotone shifts


Links
-----

UC Berkeley's Statistics 240: Nonparametric and Robust Methods.

* `2015 course
  website <http://www.stat.berkeley.edu/~johann/240spring15/index.html>`_
* `Philip Stark's lecture
  notes <http://www.stat.berkeley.edu/~stark/Teach/S240/Notes/index.htm>`_

"Permutation Tests for Complex Data: Theory, Applications and Software"
by Fortunato Pesarin, Luigi Salmaso


* `Publisher's
  website <http://www.wiley.com/WileyCDA/WileyTitle/productCd-0470516410.html>`_
* `Supplementary Material (i.e., code and
  data) <http://www.wiley.com/legacy/wileychi/pesarin/material.html>`_
* `NPC test code <http://static.gest.unipd.it/~salmaso/NPC_TEST.htm>`_

"Stochastic Ordering and ANOVA: Theory and Applications with R"
by Basso D., Pesarin F., Salmaso L., Solari A.

* `R code <http://static.gest.unipd.it/~salmaso/web/springerbook.htm>`_
"""

import os.path as _osp
import importlib as _imp
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
    _imp.import_module('nose')
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
    def _test(doctest=False, verbose=False, dry_run=False, run_all=True):
        """Run all unit tests."""
        import nose
        args = ['', pkg_dir, '--exe', '--ignore-files=^_test']
        if verbose:
            args.extend(['-v', '-s'])
        if dry_run:
            args.extend(['--collect-only'])
        if not run_all:
            args.extend(['-A', 'not slow'])
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
