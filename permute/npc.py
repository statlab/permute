from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.stats import norm


# Combining functions

def fisher(pvalues):
    '''
    Apply Fisher's combining function
    
    .. math:: \-2\sum_i \log(p_i)
    
    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine
        
    Returns
    -------
    float
        Fisher's combined test statistic
    '''
    return -2*np.log(np.prod(pvalues))


def liptak(pvalues):
    '''
    Apply Liptak's combining function
    
    .. math:: \\sum_i \Phi^{-1}(1-p_i)
    
    where $\Phi^{-1}$ is the inverse CDF of the standard normal distribution.
    
    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine
        
    Returns
    -------
    float
        Liptak's combined test statistic
    '''
    return np.sum(norm.ppf(1-pvalues))


def tippett(pvalues):
    '''
    Apply Tippett's combining function
    
    .. math:: \\max_i \{1-p_i\}
    
    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine
        
    Returns
    -------
    float
        Tippett's combined test statistic
    '''
    return np.max(1-pvalues)
