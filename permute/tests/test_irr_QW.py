"""
Unit Tests for irr.py
"""
import pytest

import numpy as np

from ..irr import (compute_ts,
                   simulate_ts_dist,
                   simulate_npc_dist)

from ..data import nsgk

def test_compute_ts():
    assert compute_ts(np.array([[1,1,1],[1,1,1]]))==1.0
    
def test_simulate_ts_dist():
    obs = simulate_ts_dist(np.array([[1,1,1],[1,1,1]]))
    assert obs['obs_ts']==1.0
    assert obs['pvalue']==1.0