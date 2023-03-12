"""
Unit Tests for qa.py
"""
import numpy as np

from ..qa import (find_duplicate_rows,
                  find_consecutive_duplicate_rows)

def test_find_duplicate_rows():
    assert np.array_equal(find_duplicate_rows(np.array([[1,2,3],[1,2,3],[2,3,4]])),np.array([[1, 2, 3]]))
    assert find_duplicate_rows(np.array([[1,2,3],[1,3,3],[2,3,4]])).size==0
    
def test_find_consecutive_duplicate_rows():
    assert np.array_equal(find_consecutive_duplicate_rows(np.array([[1,2,3],[1,2,3],[2,3,4]])),np.array([[1, 2, 3]]))
    assert find_consecutive_duplicate_rows(np.array([[1,2,3],[1,3,3],[2,3,4]])).size==0
    assert find_consecutive_duplicate_rows(np.array([[1,2,3],[2,3,4],[1,2,3]])).size==0