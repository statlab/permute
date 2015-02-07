import permute.data as data
from numpy.testing import assert_equal


def test_kenya():
    """ Test that "Kenya" data can be loaded. """
    kenya = data.kenya()
    assert_equal(kenya.shape, (16, 3))
