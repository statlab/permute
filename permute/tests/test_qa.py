from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_equal

from ..qa import (find_duplicate_rows,
                   find_consecutive_duplicate_rows)


def test_find_duplicate_rows():
    x = np.array([[1, 2, 3],
                  [2, 3, 1],
                  [1, 2, 3]])
    res1 = find_duplicate_rows(x)
    res2 = find_duplicate_rows(x, as_string=True)
    assert_equal(res1, np.array([[1, 2, 3]]))
    assert_equal(res2, ['1,2,3'])


def test_find_consecutive_duplicate_rows():
    x = np.array([[1, 2, 3],
                  [1, 2, 3],
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 2, 3]])
    res1 = find_consecutive_duplicate_rows(x)
    res2 = find_consecutive_duplicate_rows(x, as_string=True)
    assert_equal(res1, np.array([[1, 2, 3],
                                 [1, 1, 1]]))
    assert_equal(res2, ['1,2,3', '1,1,1'])
