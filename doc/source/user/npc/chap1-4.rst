Chapter 1-4 examples
====================

.. plot::
    :context:
    :nofigs:

    >>> import permute.data as data
    >>> from permute.core import one_sample, two_sample
    >>> from permute.utils import get_prng

    >>> prng = get_prng(2019426)
    >>> ipat = data.ipat()
    >>> one_sample(ipat.ya, ipat.yb, reps = 10**5, stat='mean', alternative='greater', seed=prng)


    >>> job = data.job()
    >>> two_sample(job.x[job.y == 1], job.x[job.y == 2], stat='mean', reps = 10**5, alternative='greater', seed=prng)

    >>> # TODO: three-sample ANOVA for the worms dataset

    >>> worms = data.worms()

    >>> testosterone = data.testosterone()

    >>> from permute.npc import npc
    >>> fly = data.fly()
    >>> n = fly.size
    >>> vars = fly.dtype.names[1:]

    >>> results = {}
    >>> for col in vars:
    >>>     sam1 = fly[col][fly.group == 0]
    >>>     sam2 = fly[col][fly.group == 1]
    >>>     if col == 'x7':
    >>>         results[str(col)] = two_sample(sam1, sam2, keep_dist=True, seed=prng)
    >>>     else:
    >>>         results[str(col)] = two_sample(sam1, sam2, keep_dist=True, alternative = 'less', seed=prng)
    >>> partial_pvalues = np.array(list(map(lambda col: results[col][0], vars)))
    >>> print(np.round(partial_pvalues, 3)
	
    >>> npc_distr = np.array(list(map(lambda col: results[col][2], vars))).T
    >>> npc_distr.shape
    (100000, 7)
    >>> alternatives = ['greater']*6 + ['less']*1
    >>> fisher = npc(partial_pvalues, npc_distr, alternatives=alternatives)
    >>> liptak = npc(partial_pvalues, npc_distr, alternatives=alternatives, combine = 'liptak')
    >>> tippett = npc(partial_pvalues, npc_distr, alternatives=alternatives, combine='tippett')
    >>> print("Fisher combined p-value:", fisher)
    >>> print("Liptak combined p-value:", liptak)
    >>> print("Tippett combined p-value:", tippett)