"""
Unit Tests for test_runs.py
"""

import math

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr
from ..runs import run

def smallSAMPTWOSIDED():
	assert_almost_equal(run([0, 1, 1, 0, 1, 0, 1, 0, 1], 10**5, "two-sided", False, None), 0.015)

def smallSAMPLESS():
	assert_almost_equal(run([0, 1, 1, 0, 1, 0, 1, 0, 1], 10**5, "less", False, None), 0.92)

def smallSAMPGREATER():
	assert_almost_equal(run([0, 1, 1, 0, 1, 0, 1, 0, 1], 10**5, "greater", False, None), 0.007)

def keepdist1():
	assert_equal(len(run([0, 1, 1, 0, 1, 0, 1, 0, 1], 10**5, "greater", True, None)[1]), 10**5)

def keepdist2():
	assert_equal(len(run([0, 1, 1, 0, 1, 0, 1, 0, 1], 10**5, "less", True, None)[1]), 10**5)

def keepdist3():
	assert_euqla(len(run([0, 1, 1, 0, 1, 0, 1, 0, 1], 10**5, "two-sided", True, None)[1]), 10**5)

def medREPSAMPTWOSIDED():
	sequence = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
	assert_equal(run(sequence, 10**5, "two-sided", False, None), 0)

def medREPSAMPLESS():
	sequence = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
	assert_equal(run(sequence, 10**5, "less", False, None), 1)

def medREPSAMPGREATER():
	sequence = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
	assert_equal(run(sequence, 10**5, "greater", False, None), 0)

def longRUNSEQGREATER():
	sequence = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	assert_almost_equal(run(sequence, 10**5, "greater", False, None), 0.29)

def longRUNSEQLESS():
	sequence = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	assert_almost_equal(run(sequence, 10**5, "less", False, None), 0.55)





	

