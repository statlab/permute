import numpy as np
import math
from numpy.random import RandomState

import pytest
from cryptorandom.cryptorandom import SHA256

from ..fastpermute import fastcompuation


def test_fastpermute():
    assert (np.allclose(fastcompuation([2, 6, 8, 0]), (-14, -5)))
    assert (np.allclose(fastcompuation([6, 4, 4, 6]), (-4, 10)))
    assert (np.allclose(fastcompuation([8, 4, 5, 7]), (3, 13)))