from ..permute import stratifiedPermutationTest
from nose.tools import assert_equal

import numpy as np
from numpy.testing import (assert_array_equal,
                           assert_almost_equal)

from numpy.testing import dec

@dec.skipif(True, "Skipping ...")
def test_stratifiedPermutationTest():
    group = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    condition = np.array(
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    response = np.array(
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        
    res = stratifiedPermutationTest(group, condition, response, iterations=1000)

