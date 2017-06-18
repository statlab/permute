"""
Unit Test for fisher_exact_test.py
"""

import unittest
from fisher_exact_test import fisher


class fisher_tests(unittest.TestCase):
	"""
	Tests for 'fisher_exact_test.py'

	Values obtained from:

	http://www.graphpad.com/quickcalcs/contingency2/

	"""

	def test_one_tail(self):
		self.assertEqual(round(fisher([[2, 3], [8, 2]], 2), 2), 0.25)
	
	def test_two_tail(self):
		self.assertEqual(round(fisher([[2, 3], [8, 2]], 1), 4), 0.1668)

	def test_one_t_alt(self):
		self.assertEqual(round(fisher([[2, 3], [6, 4]], 2), 3), 0.608)

	def test_two_t_alt(self):
		self.assertEqual(round(fisher([[2, 3], [6, 4]], 1), 3), 0.427)
		

if __name__ == '__main__':
	unittest.main()
