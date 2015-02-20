# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import scipy.misc
import scipy.random

def compute_ts(ratings):
    """
    Compute the test statistic
    .. math:: \rho_s \equiv
        \frac{1}{N_s {R \choose 2}} \sum_{i=1}^{N_s} \sum_{r=1}^{R-1}
            \sum_{v=r+1}^R 1(L_{s,i,r} = L_{s,i,v})

    Parameters
    ----------
    ratings: array_like
             Input array of dimension [R, Ns]

    Returns
    -------
    rho_s: float
           concordance of the ratings, where perfect concordance is 1.0
    """
    R, Ns = ratings.shape
    tmp = ratings.sum(0)
    counts = scipy.misc.comb(tmp, 2) + scipy.misc.comb(R-tmp, 2)
    rho_s = counts.sum()/(Ns*scipy.misc.comb(R, 2))
    return rho_s

def permute_rows(ratings):
    """
    Permute elements of each row in the ratings matrix except the top row.
    Each row corresponds to the ratings given by a single rater; columns correspond to items rated.
    Parameters
    ----------
    ratings: array_like
             Input array of dimension [R, Ns]

    Returns
    -------
    True

    Action
    ------
    Permutes the elements of each row of <ratings> in place; leaves the top row unchanged
    """
    np.apply_along_axis(np.random.shuffle, axis=1, arr=ratings)
    return True