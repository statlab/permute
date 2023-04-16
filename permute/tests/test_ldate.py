#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests for functions in ldate

@author: wangqiyu
"""

import numpy as np
import pytest

from ldate import *

def test_filter_table():
    assert filter_table([2,2,2,2],4,0,2,2)
    assert not filter_table([2,2,2,2],5,0,1,2)
    
def test_N_generator():
    Nt = N_generator(4,1,1,1,1)
    assert [2,1,1,0] in Nt
    assert [1,2,1,0] in Nt
    assert [1,1,2,0] in Nt
    assert [2,0,0,2] in Nt
    
def test_sim_obs(): 
    for nt in sim_obs([10,5,5,10],12,3,np.random.RandomState(5)):
        assert filter_table([10,5,5,10],nt[0],nt[1],nt[2],nt[3])
        
def test_T():
    obs = T([10,5,5,10],[11, 7, 5, 7])
    np.testing.assert_almost_equal(obs,0.1944444,decimal=3)
    
def test_stat_consist():   
    assert stat_consist([1,1,1,1],[1,1,1,1],T=T,rep=10,rng=np.random.RandomState())==1
    
def test_ate_ci():    
    assert ate_ci([1,1,1,1],rep=10) == (-0.25,0.25)
    
