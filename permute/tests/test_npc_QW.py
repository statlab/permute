"""
Unit Tests for npc.py
"""

import pytest

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm

from ..npc import (fisher,
                   liptak,
                   tippett,
                   inverse_n_weight,
                   npc,
                   check_combfunc_monotonic,
                   sim_npc,
                   westfall_young,
                   adjust_p,
                   fwer_minp,
                   randomize_group,
                   Experiment)

def test_fisher():
    obs = fisher(np.array([0.1,0.2,0.3]))
    expected = 10.231991619508165
    np.testing.assert_almost_equal(obs, expected, decimal=3)
    
def test_liptak():
    obs = liptak(np.array([0.2,0.4,0.6]))
    expected = 0.8416212335729143
    np.testing.assert_almost_equal(obs, expected, decimal=3)
    
def test_tippett():
    assert tippett(np.array([0.5,0.7,0.9]))==0.5
    
def test_inverse_n_weight():
    obs = inverse_n_weight(np.array([0.3,0.4,0.5]),np.array([10,10,10]))
    expected = -0.37947331922020555
    np.testing.assert_almost_equal(obs, expected, decimal=3)
    
def test_check_combfunc_monotonic():
    assert check_combfunc_monotonic(np.array([0.3,0.4,0.5]),tippett)
    assert check_combfunc_monotonic(np.array([0.6,0.8,1.0]),liptak)
    assert check_combfunc_monotonic(np.array([0.25,0.50,0.75]),fisher)
    
def test_npc():
    obs = npc(np.array([0.3,0.4,0.5]),np.array([[1,1,1],[2,2,2]]))
    expected = 0.3333333333333333
    np.testing.assert_almost_equal(obs, expected, decimal=3)

    
def test_adjust_p():
    obs=adjust_p(np.array([0.3,0.4,0.5]))
    np.testing.assert_almost_equal(obs[0], 0.9, decimal=3)
    np.testing.assert_almost_equal(obs[1], 0.9, decimal=3)
    np.testing.assert_almost_equal(obs[2], 0.9, decimal=3)
