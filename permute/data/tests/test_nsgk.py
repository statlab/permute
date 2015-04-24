import permute.data as data
import numpy as np
from numpy.testing import assert_equal


def test_nsgk():
    """ Test that "NSGK" data can be loaded. """
    nsgk = data.nsgk()
    assert_equal(nsgk.shape, (40, 183, 8, 10))
    assert_equal(nsgk.dtype, np.dtype('int64'))
    assert_equal(nsgk.sum(), 24713)
