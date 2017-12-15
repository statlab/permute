from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import math
from scipy.stats import linregress

from utils import get_prng

def reg_coeff(x, y, reps=10**5, cl=0.95, alternative="one-sided", seed=None):
    r"""
    Testing the coefficients of a linear regression. The assumption of the linear regression
    is that 

    Parameters
    ----------
    x : array-like
        list of observations
    N : int
        population size
    cl : float in (0, 1)
        the desired confidence level. Default 0.95.
    lower_bound : scalar or array-like
        lower bound for the observations
    upper_bound : scalar or array-like
        upper bound for the observations
    alternative : {'greater', 'less', 'two-sided'}
       alternative hypothesis to test (default: 'greater')
    
    Returns
    -------
    tuple
       the confidence limits
    """
    true_slope = abs(linregress(x, y)[0])

    prng = get_prng(seed)
    slopes = []
    for _ in range(reps):
        shuffle_x = np.random.permutation(x)
        slope = linregress(shuffle_x, y)[0]
        slopes.append(slope)

    p_value = sum(abs(np.array(slopes)) > true_slope)/float(reps)

    return p_value  
    
