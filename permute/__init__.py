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

__version__ = "0.2"


try:
    _imp.import_module('pytest')
except ImportError:
    def test(verbose=False):
        """This would run all unit tests, but pytest couldn't be
        imported so the test suite can not run.
        """
        raise ImportError("Could not load pytest. Unit tests not available.")

    def _doctest(verbose=False):
        """This would run all doc tests, but pytest couldn't be
        imported so the test suite can not run.
        """
        raise ImportError("Could not load pytest. Doctests not available.")
else:
    def test(doctest=False, verbose=False, dry_run=False, run_all=True):
        """Run all unit tests."""
        import pytest

        pytest_args = ["-l"]

        if verbose:
            pytest_args += ["-v"]

        if dry_run:
            pytest_args += ["--collect-only"]

        if run_all:
            pytest_args += ["--runslow"]

        if doctest:
            pytest_args += ["--doctest-modules"]

        pytest_args += ["--pyargs", "permute"]

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code
    
        return code == 0
