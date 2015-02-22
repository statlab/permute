"""
There are $S$ strata (in this case, each stratum is a video).  There are $N_s$ 
items in stratum $s$ (in this case, an item is corresponds to a 30-second
snippet of the video). There are $N = \sum_{s=1}^S N_s$ items in all.

There are $C$ non-exclusive categories to which each of the $N$ items might 
belong; an item might belong to none of the categories (the categories are
types of behavior that a child might exhibit in the 30-second snippet of
video). That is, each item might be "labeled" with any of the $2^C$ subsets
of the $C$ labels, including the empty set.

There are $R$ "raters," each of whom labels each of the $N$ items with zero 
or more elements of $C$.

Define $L_{s,i,c,r} = 1$, if rater $r$ assigns label $c$ to item $i$ in stratum 
$s$; and $L_{s,i,c,r} = 0$ if not.

We observe $\{ L_{s,i,c,r} \}$ for $s=1...S$;  $i=1, ..., N_i$; $c=1, ..., C$; 
and $r=1, ..., R$.

We want to know whether the categorizations are "reliable," in the sense that 
agreement among the raters is higher than would be expected "by chance."  The
reliability of each category $c$ is of interest, rather than an overall rating
for all $C$ categories.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

import scipy.misc

def compute_ts(ratings):
    """
    Compute the test statistic

    .. math:: \\rho_s \equiv \\frac{1}{N_s {R \choose 2}} \sum_{i=1}^{N_s}
              \sum_{r=1}^{R-1} \sum_{v=r+1}^R 1(L_{s,i,r} = L_{s,i,v})

    Parameters
    ----------
    ratings: array_like
             Input array of dimension [R, Ns]
             Each row corresponds to the ratings given by a single rater;
             columns correspond to items rated.

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


def simulate_ts_dist(ratings, obs_ts = None, iter=10000, keep_dist = False):
    """
    Simulates the permutation distribution of the irr test statistic for a matrix of
    ratings <ratings>

    If obs_ts is not null, computes the reference value of the test statistic before
    the first permutation. Otherwise, uses the value obs_ts for comparison.

    If <keep_dist>, return the distribution of values of the test statistic; otherwise,
    return only the number of permutations for which the value of the irr test statistic is
    at least as large as obs_ts.

    Parameters
    ----------
    ratings : array_like
              Input array of dimension [R, Ns]

    obs_ts : float
             if None, obs_ts is calculated as the value of the test statistic for the
             original data

    iter : integer
           number of random permutation of the elements of each row of ratings

    keep_dist : bool
                flag for whether to store and return the array of values of the irr
                test statistic


    Returns
    -------
    out : {obs_ts, geq, iter, dist}
    obs_ts : observed value of the test statistic for the input data, or the input value
             of obs_ts if obs_ts was given as input
    geq : number of iterations for which the test statistic was greater than or equal to
          obs_ts
    iter : iter
    dist : if <keep_dist>, the array of values of the irr test statistic from the iter
           iterations.  Otherwise, null.
    """
    r = ratings.copy()

    if obs_ts is None:
        obs_ts = compute_ts(r)


    if keep_dist:
        dist = np.zeros(iter)
        dist = [compute_ts(np.apply_along_axis(np.random.permutation, axis=1, arr=r)) for i in range(iter)]
        geq = np.sum(dist >= obs_ts)
    else:
        dist = None
        geq = 0
        for i in range(iter):
            geq += compute_ts(np.apply_along_axis(np.random.permutation, axis=1, arr=r)) >= obs_ts
    return {"obs_ts": obs_ts, "geq": geq, "iter": iter, "dist": dist}
