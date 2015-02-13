# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import scipy.misc

def compute_ts(ratings):
    """
    Compute the test statistic
    .. math:: \rho_s \equiv \frac{1}{N_s {R \choose 2}} \sum_{i=1}^{N_s} \sum_{r=1}^{R-1} \sum_{v=r+1}^R 1(L_{s,i,r} = L_{s,i,v})

    Parameters
    ----------
    a : array_like
        Input array of dimension [R, Ns]

    Returns
    -------
    rho_s: float
    """
    R, Ns = ratings.shape
    tmp = ratings.sum(0)
    counts = scipy.misc.comb(tmp, 2) + scipy.misc.comb(R-tmp, 2)
    rho_s = counts.sum()/(Ns*scipy.misc.comb(R, 2))
    return rho_s
