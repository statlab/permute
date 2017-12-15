import math
import numpy as np
import scipy as sp

def kaplan_wald_CI(x, level=0.99, lo=0, hi=float('inf'), gamma=0.95, xtol=1.e-32):
    '''
       Calculates the confidence interval for the mean of a nonnegative random variable
       using the Kaplan-Wald method.
       
       Parameters
       ----------
       x : array-like
           your input sample
       level : float in (0, 1)
           indicates your desired confidence level
       lo : float in [0, float('inf')]
            the lower bound of your random variable, must be nonnegative
       hi : float in (hi, float('inf'))
            the upper bound of your random variable, optional
       
       Returns
       -------
       tuple
           The estimated confidence level. If the upper bound if not specified,
           only the lower bound of the confidence interval will be returned.

       Notes
       -----
       gamma : float in (0, 1)
               the variable introduced in Kaplan-Wald method to hedge against small values
       xtol : float
           Tolerance in brentq
    '''
    alpha = 1.0 - level
    if not (0 < level < 1):
        raise ValueError('CI level must be between 0 and 1.')
    if any(x < 0):
        raise ValueError('Data x must be nonnegative.')
    if lo > hi or any(x < lo) or any(x > hi):
        raise ValueError('Data x is not in the specified range.')
    def find_t(data, start):
        f = lambda t: (np.max(np.cumsum(np.log(gamma*data/t + 1-gamma))) + np.log(alpha))
        start, end = start, np.mean(data)
        if f(start) * f(end) <= 0.0:
            return sp.optimize.brentq(f, start, end, xtol=xtol)
        else:
            return start
    lo = find_t(x, lo + xtol)
    if hi != float('inf'):
        hi = find_t(hi - x, xtol) + np.mean(x)
    return (lo, hi)


def sprt_proportion(x, p0, p1, alpha, beta, start=1, plot=False):
    '''
       The model aims to determine the quality of a batch of products by minimal sampling.
       The idea is to sample the batch sequentially until a decision can be made whether
       the batch conforms to specification and can be accepted or that it should be rejected.
       
       Parameters
       ----------
       x : array-like. your input sample.
           each value should indicate success or failure. 
       alpha: float in [0, 1]
              type I error
       beta: float in [0, 1]
             type II error
       p0: float in [0, 1]. proportion to accept null hypothesis
           the proportion of positives below which
           a decision that the proportion is null can be made
       p1: float in [0, 1]. proportion to reject null hypothesis
           the proportion of positives above which
           a decision that the proportion is null can be made 
       
       Returns
       -------
       tuple
           tuple[0] can be'Reject' the null, 'Accept' the null,
           or 'Unknown' based on the data
           tuple[1] is the log likelihood ratio for SPRT when a
           decision is made or running out of data
           tuple[2] is the index of the trail where a decision is made

       Notes
       -----
       start
           If the last batch of data gives unknown decision,
           one can feed the tuple[1] into the param start to continue the SPRT.
    '''
    if not (0 <= p0 <= 1 and 0 <= p1 <= 1):
        raise ValueError('Proportion must lie in [0, 1]')
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError('Error control param must lie in (0, 1)')
    lr = start
    s_fac = math.log(p1 / p0)
    f_fac = math.log((1.0-p1) / (1.0-p0))
    s_bnd = math.log((1-beta) / alpha)
    f_bnd = math.log(beta / (1-alpha))
    decision = 'Unknown'
    for idx, trail in enumerate(x, 1):
        trail = int(trail)
        if trail != 0 and trail != 1:
            raise ValueError('every trail in the data must be a truthy/falsy value.')
        lr += s_fac if trail == 1 else f_fac
        if lr >= s_bnd:
            decision = 'Reject'
            break
        elif lr <= f_bnd:
            decision = 'Accept'
    return (decision, lr, idx)
