from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import math

def hoeffding_conf_int(x, N, lower_bound, upper_bound, cl=0.95, alternative="one-sided"):
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
    lower_bound : scalar or array-like
        lower bound for the observations
    upper_bound : scalar or array-like
        upper bound for the observations
    alternative : {'one-sided', 'two-sided'}
       alternative hypothesis to test (default: 'one-sided')
    
    Returns
    -------
    tuple
       the confidence limits
    """ 
    #Check if Bounds are Scalar or Array-Like
    if (isinstance(lower_bound, float) or isinstance(lower_bound, int)) and (isinstance(upper_bound, float) or isinstance(upper_bound, int)):
        if max(x) > upper_bound or min(x) < lower_bound:
            raise ValueError("x values not contained in bounds")
        tau_sq = N*((upper_bound - lower_bound)**2)
        max_upper = upper_bound
        min_lower = lower_bound
    elif isinstance(lower_bound, np.ndarray) and isinstance(lower_bound, np.ndarray):
        #Makes sure the bounds are valid, same length, makes sense, and values are in the bounds
        if len(lower_bound) != N or len(upper_bound) != N:
            raise ValueError("Bad Bound Input Length")
        if np.sum(upper_bound >= lower_bound) != N:
            raise ValueError("Invalid Upper and Lower bounds")
        if np.sum((x <= upper_bound)) != N or np.sum((x >= lower_bound)) != N:
            raise ValueError("x values not contained in bounds")
        else:
            tau_sq = np.sum((upper_bound - lower_bound)**2)
        max_upper = np.mean(upper_bound)
        min_lower = np.mean(lower_bound)
    else:
        raise ValueError("Invalid Upper and Lower Bounds")

    x_bar = np.mean(x)

    if alternative == "one-sided":
        hCrit = np.sqrt(-math.log(1-cl)*tau_sq/(2*N**2))
        ci_low = x_bar - hCrit
        ci_upp = max_upper

    if alternative == "two-sided":
        hCrit = np.sqrt(-math.log((1-cl)/2)*tau_sq/(2*N**2))
        ci_low, ci_upp = x_bar - hCrit, x_bar + hCrit

    #Truncated Bounds of the Hoeffding Confidence Interval
    if ci_upp > max_upper:
        ci_upp = max_upper
    if ci_low < min_lower:
        ci_low = min_lower

    return ci_low, ci_upp
