Chapter 1-4 examples
====================

IPAT Data
---------

This example is shown in Chapter 1.9, page 33-34.

.. plot::
    :context:
    :nofigs:

    >>> import permute.data as data
    >>> from permute.core import one_sample, two_sample
    >>> from permute.ksample import k_sample, bivariate_k_sample
    >>> from permute.utils import get_prng
    >>> import numpy as np

    >>> prng = get_prng(2019426)
    >>> ipat = data.ipat()
    >>> res = one_sample(ipat.ya, ipat.yb, stat='mean', alternative='greater', seed=prng)
    >>> print("P-value:", round(res[0], 4))
    P-value: 0.0002

Job Satisfaction Data
---------------------

This example is shown in Chapter 1.10.3, page 41-42.

.. plot::
    :context:
    :nofigs:

    >>> job = data.job()
    >>> res = two_sample(job.x[job.y == 1], job.x[job.y == 2], stat='mean', reps = 10**5, alternative='greater', seed=prng)
    >>> print("P-value:", round(res[0], 4))
    P-value: 0.0003

Worms Data
----------

This example is shown in Chapter 1.11.12, page 47-48.

.. plot::
    :context:
    :nofigs:

    >>> worms = data.worms()
    >>> res = k_sample(worms.x, worms.y, stat='one-way anova', seed=prng)
    >>> print("ANOVA p-value:", round(res[0], 4))
    ANOVA p-value: 0.011

Testosterone Data
-----------------

This example is shown in Cbapter 2.6.1, page 92-93.

.. plot::
    :context:
    :nofigs:

    >>> testosterone = data.testosterone()
    >>> x = np.hstack(testosterone.tolist())
    >>> group1 = np.hstack([[i]*5 for i in range(len(testosterone))])
    >>> group2 = np.array(list(range(5))*len(testosterone))
    >>> print(len(group1), len(group2), len(x))
    55 55 55
    >>> res = bivariate_k_sample(x, group1, group2, reps=5000, seed=prng) 
    >>> print("ANOVA p-value:", round(res[0], 4))
    ANOVA p-value: 0.0078

Fly Data
--------

This example is shown in Chapter 4.6, page 253.

.. plot::
    :context:
    :nofigs:

    >>> from permute.npc import npc
    >>> fly = data.fly()
    >>> vars = fly.dtype.names[1:]
    >>> results = {}
    >>> for col in vars:
    >>>     sam1 = fly[col][fly.group == 0]
    >>>     sam2 = fly[col][fly.group == 1]
    >>>     if col == 'x7':
    >>>         results[str(col)] = two_sample(sam1, sam2, keep_dist=True, seed=prng, plus1=True, reps=10**4)
    >>>     else:
    >>>         results[str(col)] = two_sample(sam1, sam2, keep_dist=True, alternative = 'less', seed=prng, plus1=True, reps=10**4)
    >>> partial_pvalues = np.array(list(map(lambda col: results[col][0], vars)))
    >>> print(np.round(partial_pvalues, 3))
    [0.022 0.212 0.    0.337 0.    0.332 0.096]

    >>> npc_distr = np.array(list(map(lambda col: results[col][2], vars))).T
    >>> npc_distr.shape
    (100000, 7)
    >>> alternatives = ['greater']*6 + ['less']*1
    >>> fisher = npc(partial_pvalues, npc_distr, alternatives=alternatives)
    >>> liptak = npc(partial_pvalues, npc_distr, alternatives=alternatives, combine = 'liptak')
    >>> tippett = npc(partial_pvalues, npc_distr, alternatives=alternatives, combine='tippett')
    >>> print("Fisher combined p-value:", fisher)
    Fisher combined p-value: 0.0
    >>> print("Liptak combined p-value:", liptak)
    Liptak combined p-value: 0.0
    >>> print("Tippett combined p-value:", tippett)
    Tippett combined p-value: 0.0
