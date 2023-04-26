import numpy as np
import math
from numpy.random import RandomState

import pytest
from cryptorandom.cryptorandom import SHA256

from ..fastpermute import fastcompuation


def test_fastpermute():
        self.assertTrue(np.allclose(find_interval(0.05, [2, 6, 8, 0])[1], [-14, -5]))
        self.assertTrue(np.allclose(find_interval(0.05, [6, 4, 4, 6])[1], [-4, 8]))
        self.assertTrue(np.allclose(find_interval(0.05, [8, 4, 5, 7])[1], [-3, 9]))