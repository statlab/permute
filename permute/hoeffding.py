from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

def hoeffding_conf_int(x, N, cl=0.95, lower_bound=min(x), upper_bound=max(x), alernative="one-sided"):

    r"""
    Confidence interval for the mean of bounded, independent observations 
	derived by inverting Hoeffding's inequality. This method uses the 
	assumption that the observations have the same bounds.

    Parameters
    ----------
    x : array-like
        list of observations
    N : int
        population size
    cl : float in (0, 1)
    	the desired confidence level. Default 0.95.
    lower_bound : float
    	lower bound for the observations
    upper_bound : float
    	upper bound for the observations
    alternative : {'greater', 'less', 'two-sided'}
       alternative hypothesis to test (default: 'greater')
    
    Returns
    -------
    tuple
       the confidence limits
    """	
	x_bar = np.mean(x)
	tau_sq = N*math.pow(upper_bound - lower_bound)
	alpha = 1-cl
	hCrit = np.sqrt(-math.log(alpha/2)*tau_sq/(2*N**2))
	ci_low, ci_upp = x_bar - hCrit, x_bar + hCrit
	return ci_low, ci_upp 