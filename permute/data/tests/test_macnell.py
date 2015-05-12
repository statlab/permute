import permute.data as data
from numpy.testing import assert_equal


def test_macnell2014():
    """ Test that "MacNell2014" data can be loaded. """
    macnell = data.macnell2014()
    assert_equal((macnell.size, len(macnell.dtype)), (43, 20))
