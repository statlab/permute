Westfall-Wolfinger "mult" data
===============================

This example comes from Chapter 5.5.

When running the two sample test, use the same initial PRNG seed in order to keep permutations correlated for each test. This is crucial for NPC.

.. nbplot::

    >>> import permute.data as data
    >>> from permute.core import two_sample
    >>> from permute.npc import fwer_minp
    >>> import numpy as np

    >>> ww = data.mult()
    >>> p1 = two_sample(ww.y1[ww.x == 0], ww.y1[ww.x == 1], alternative="two-sided", seed=40929102, keep_dist=True, reps=5000)
    >>> p2 = two_sample(ww.y2[ww.x == 0], ww.y2[ww.x == 1], alternative="two-sided", seed=40929102, keep_dist=True, reps=5000)
    >>> p3 = two_sample(ww.y3[ww.x == 0], ww.y3[ww.x == 1], alternative="two-sided", seed=40929102, keep_dist=True, reps=5000)
    
    >>> pvalues = np.array([p1[0], p2[0], p3[0]])
    >>> distr = np.abs(np.vstack([p1[2], p2[2], p3[2]]).T)
    >>> pvalues_adj_fisher = fwer_minp(pvalues, distr, combine="fisher")
    >>> pvalues_adj_liptak = fwer_minp(pvalues, distr, combine="liptak")
    >>> pvalues_adj_tippett = fwer_minp(pvalues, distr, combine="tippett")


::

    print("Adjusted p-values \nFisher:", pvalues_adj_fisher, "\nLiptak:", pvalues_adj_liptak, "\nTippett:", pvalues_adj_tippett)
    Adjusted p-values 
    Fisher: [0.08758248 0.04819036 0.0219956 ] 
    Liptak: [0.08758248 0.0459908  0.02339532] 
    Tippett: [0.08758248 0.06258748 0.02359528]
