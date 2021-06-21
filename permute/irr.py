r"""
A stratified permutation test for multi-rater inter-rater reliability.

There are $S$ strata.
There are $N_s$ items in stratum $s$.
There are $N = \sum_{s=1}^S N_s$ items in all.

There are $C$ non-exclusive categories to which each of the $N$ items might
belong; an item might belong to none of the categories.
That is, each item might be "labeled" with any of the $2^C$ subsets
of the $C$ labels, including the empty set.

There are $R$ "raters," each of whom labels each of the $N$ items with zero
or more elements of $C$.

Define $L_{s,i,c,r} = 1$, if rater $r$ assigns label $c$ to item $i$ in
stratum $s$; and $L_{s,i,c,r} = 0$ if not.

We observe $\{ L_{s,i,c,r} \}$ for $s=1...S$;  $i=1, ..., N_s$;
$c=1, ..., C$; and $r=1, ..., R$.

We want to know whether the categorizations are "reliable," in the sense
that agreement among the raters is higher than would be expected
"by chance."  The reliability of each category $c$ is of interest,
rather than an overall rating for all $C$ categories.

Fix $c$, since we are considering only one category at a time.

The null hypothesis for category $c$ is that, for each rater $r$, and each
stratum $s$, the values $\{ L_{s,i,c,r} \}$ are exchangeable; that for
each rater $r$, the values $\{ L_{s,i,c,r} \}$ for different strata $s$
are independent; and that the values are independent across raters.

Our test conditions on the sets of labels each rater assigns
within each stratum, but not on the items to which those labels are
assigned.  The null distribution involves permuting the assignments each
given rater makes of category $c$ to items within each stratum $s$,
permuting independently across across raters and across strata.

The test statistic within stratum $s$ is

.. math:: \rho_s \equiv \frac{1}{N_s {R \choose 2}} \sum_{i=1}^{N_s}
              \sum_{r=1}^{R-1} \sum_{v=r+1}^R 1(L_{s,i,r} = L_{s,i,v})
              = \frac{1}{N_s R(R-1)} \sum_{i=1}^{N_s}
                (y_{si}(y_{si}-1) + (R-y_{si})(R-y_{si}-1)).

That is, within each stratum, we count the number of concordant pairs of
assignments. If all $R$ raters agree whether item $i$ in stratum $s$
belongs to category $c$, that contributes a term ${R \choose 2}$ to the
sum.  If only half agree, the term for item $i$ contributes
$2 {N/2 \choose 2}$ to the sum.  The normalization makes perfect
agreement within stratum $s$ correspond to $\rho_s = 1$.

To combine the results across strata to get an overall p-value, we could
use any of the methods we've discussed, or the NPC (nonparametric
combination of test) methods described in Pesarin and Salmaso, based on
the p-values in different strata.  For instance, Fisher's combination
statistic is

.. math:: \lambda = - \sum_{s=1}^S w_s \log \hat{p}_s,

where the nonnegative weights $\{w_s\}$ are chosen in some sensible manner
(e.g., $w_s = N_s^{-1/2}$ would be reasonable).
"""


import numpy as np

from .utils import get_prng, permute_rows
from .npc import inverse_n_weight, npc


def compute_ts(ratings):
    r"""
    Compute the test statistic

    .. math:: \rho_s \equiv \frac{1}{N_s {R \choose 2}} \sum_{i=1}^{N_s}
              \sum_{r=1}^{R-1} \sum_{v=r+1}^R 1(L_{s,i,r} = L_{s,i,v})
              = \frac{1}{N_s R(R-1)} \sum_{i=1}^{N_s}
                (y_{si}(y_{si}-1) + (R-y_{si})(R-y_{si}-1)).

    Parameters
    ----------
    ratings : array_like
        Input array of dimension [R, Ns]
        Each row corresponds to the ratings given by a single rater;
        columns correspond to items rated.

    Returns
    -------
    rho_s : float
        concordance of the ratings, where perfect concordance is 1.0
    """
    R, Ns = ratings.shape
    y = ratings.sum(0)
    counts = y * (y - 1) + (R - y) * (R - y - 1)
    rho_s = counts.sum() / (Ns * R * (R - 1))
    return rho_s


def simulate_ts_dist(ratings, obs_ts=None, num_perm=10000,
                     keep_dist=False, seed=None, plus1=True):
    r"""
    Simulates the permutation distribution of the irr test statistic for
    a matrix of ratings ``ratings``

    If ``obs_ts`` is not ``None``, computes the reference value of the test
    statistic before the first permutation. Otherwise, uses the value
    ``obs_ts`` for comparison.

    If ``keep_dist``, return the distribution of values of the test statistic;
    otherwise, return only the number of permutations for which the value of
    the irr test statistic is at least as large as ``obs_ts``.

    Parameters
    ----------
    ratings : array_like
              Input array of dimension [R, Ns]
    obs_ts : float
             if None, ``obs_ts`` is calculated as the value of the test
             statistic for the original data
    num_perm : int
           number of random permutation of the elements of each row of ratings
    keep_dist : bool
                flag for whether to store and return the array of values of
                the irr test statistic
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator
    plus1 : bool
        flag for whether to add 1 to the numerator and denominator of the
        p-value based on the empirical permutation distribution.
        Default is True.

    Returns
    -------
    dict
        A dictionary containing:

        obs_ts : int
            observed value of the test statistic for the input data, or
            the input value of ``obs_ts`` if ``obs_ts`` was given as input
        geq : int
            number of iterations for which the test statistic was greater
            than or equal to ``obs_ts``
        num_perm : int
            number of permutations
        pvalue : float
            geq / num_perm
        dist : array-like
            if ``keep_dist``, the array of values of the irr test statistic
            from the ``num_perm`` iterations.  Otherwise, ``None``.
    """
    r = ratings.copy()
    prng = get_prng(seed)

    if obs_ts is None:
        obs_ts = compute_ts(r)

    if keep_dist:
        dist = np.zeros(num_perm)
        for i in range(num_perm):
            r = permute_rows(r, prng)
            dist[i] = compute_ts(r)
        geq = np.sum(dist >= obs_ts)
    else:
        dist = None
        geq = 0
        for i in range(num_perm):
            r = permute_rows(r, prng)
            geq += (compute_ts(r) >= obs_ts)
    return {"obs_ts": obs_ts,
            "geq": geq,
            "num_perm": num_perm,
            "pvalue": (geq+plus1) / (num_perm+plus1),
            "dist": dist}


def simulate_npc_dist(perm_distr, size, obs_ts=None,
                      pvalues=None, plus1=True):
    r"""
    Simulates the permutation distribution of the combined NPC test
    statistic for S matrices of ratings ``ratings`` corresponding to
    S strata. The distribution comes from applying ``simulate_ts_dist``
    to each of the S strata.

    If obs_ts is not null, computes the reference value of the test
    statistic before the first permutation. Otherwise, uses the value
    ``obs_ts`` for comparison.

    If ``keep_dist``, return the distribution of values of the test
    statistic; otherwise, return only the number of permutations
    for which the value of the irr test statistic is at least
    as large as ``obs_ts``.

    Parameters
    ----------
    perm_distr : array_like
        Input array of dimension [B, S]
        Column s is the permutation distribution of $\rho_s$,
        for s=1,...,S
    size : array_like
        Input array of dimension S
        Each entry corresponds to the number of items, Ns,
        in the s-th stratum.
    obs_ts : array_like
        Optional input array of dimension S
        The s-th entry is $\rho_s$, the concordance for the s-th stratum.
        If not input, pvalues must be specified.
    pvalues : array_like
        Optional input array of dimension S
        The s-th entry is the p-value corresponding to $\rho_s$,
        the concordance for the s-th stratum.
        If not input, obs_ts must be specified.
    plus1 : bool
        flag for whether to add 1 to the numerator and denominator of the
        p-value based on the empirical permutation distribution.
        Default is True.

    Returns
    -------
    dict
        A dictionary containing:

        obs_npc : float
            observed value of the combined test statistic for the input data, or
            the input value of ``obs_ts`` if ``obs_ts`` was given as input
        pvalue : float
            A single p-value for the global test. The number of times that obs_npc
            was at least as extreme as the distribution of combined IRR statistics.
        num_perm : int
            number of permutations
    """
    if (obs_ts is None) and (pvalues is None):
        raise ValueError('You must input either obs_ts or pvalues')

    (B, S) = perm_distr.shape
    combine_func = lambda p: inverse_n_weight(p, size)

    if pvalues is None:
        pvalues = np.zeros(S)
        for j in range(S):
            pvalues[j] = (np.sum(perm_distr[:, j] >= obs_ts[j])+plus1)/(B+plus1)

    obs_npc = combine_func(pvalues)
    res = npc(pvalues, perm_distr, combine_func)
    return {"obs_npc": obs_npc,
            "pvalue": res,
            "num_perm": B}
