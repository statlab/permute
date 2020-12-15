import permute.data as data
import numpy as np
from numpy.testing import assert_equal


def test_nsgk():
    """ Test that "NSGK" data can be loaded. """
    nsgk = data.nsgk()
    assert len(nsgk) == 183
    assert len(nsgk[0]) == 8
    assert nsgk[0][0].shape == (10, 36)
    assert nsgk[0][5].shape == (10, 35)
    assert nsgk[0][0].dtype == np.dtype('int32')
    assert nsgk[6][2].dtype == np.dtype('int32')
    yy = [x.sum() for y in nsgk for x in y]
    assert np.array(yy).sum() == 24713
