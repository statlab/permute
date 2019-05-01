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
    >>> ipat_res = one_sample(ipat.ya, ipat.yb, stat='mean', alternative='greater', seed=prng)
    >>> print("P-value:", round(ipat_res[0], 4))
    P-value: 0.0002

Job Satisfaction Data
---------------------

This example is shown in Chapter 1.10.3, page 41-42.

.. plot::
    :context:
    :nofigs:

    >>> job = data.job()
    >>> job_res = two_sample(job.x[job.y == 1], job.x[job.y == 2], stat='mean', reps = 10**5, alternative='greater', seed=prng)
    >>> print("P-value:", round(job_res[0], 4))
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
    ANOVA p-value: 0.0107

Testosterone Data
-----------------

This example is shown in Chapter 2.6.1, page 92-93.

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
    ANOVA p-value: 0.0002


Massaro-Blair Data
------------------
This example is shown in Chapter 4.6, page 240.

.. plot::
    :context:
    :nofigs:

    >>> from permute.npc import npc
    >>> mb = data.massaro_blair()
    >>> sam1 = mb.y[mb.group == 1]
    >>> sam2 = mb.y[mb.group == 2]
    >>> first_moment = two_sample(sam1, sam2, alternative='two-sided', reps=5000, keep_dist=True, seed=42)
    >>> second_moment = two_sample(sam1**2, sam2**2, alternative='two-sided', reps=5000, keep_dist=True, seed=423)
    >>> partial_pvalues = np.array([first_moment[0], second_moment[0]])
    >>> print("Partial p-values:", round(first_moment[0], 3), round(second_moment[0], 3))
    Partial p-values: 0.017 0.009

    >>> npc_distr = np.vstack([first_moment[2], second_moment[2]]).T
    >>> global_p = npc(partial_pvalues, npc_distr, alternatives='two-sided')
    >>> print("Global p-value:", round(global_p, 4))
    Global p-value: 0.0018

Fly Data
--------

This example is shown in Chapter 4.6, page 253.

::

    fly = data.fly()
    vars = fly.dtype.names[1:]
    results = {}
    for col in vars:
        sam1 = fly[col][fly.group == 0]
        sam2 = fly[col][fly.group == 1]
        if col == 'x7':
            results[str(col)] = two_sample(sam1, sam2, keep_dist=True, seed=prng, plus1=True, reps=10**4)
        else:
            results[str(col)] = two_sample(sam1, sam2, keep_dist=True, alternative = 'less', seed=prng, plus1=True, reps=10**4)
    partial_pvalues = np.array(list(map(lambda col: results[col][0], vars)))
    print(np.round(partial_pvalues, 3))
    [0.022 0.212 0.    0.337 0.    0.332 0.096]

    npc_distr = np.array(list(map(lambda col: results[col][2], vars))).T
    npc_distr.shape
    000, 7)
    alternatives = ['greater']*6 + ['less']*1
    fisher = npc(partial_pvalues, npc_distr, alternatives=alternatives)
    liptak = npc(partial_pvalues, npc_distr, alternatives=alternatives, combine = 'liptak')
    tippett = npc(partial_pvalues, npc_distr, alternatives=alternatives, combine='tippett')
    print("Fisher combined p-value:", fisher)
    er combined p-value: 0.0
    print("Liptak combined p-value:", liptak)
    ak combined p-value: 0.0
    print("Tippett combined p-value:", tippett)
    Tippett combined p-value: 0.0


Post-hoc conditional power analysis
-----------------------------------

These examples come from Chapter 3.2.1, pages 139-141.

:: 

    # IPAT data
    alpha = 0.01
    prng = get_prng(78943501)
    effect_est = ipat_res[1]
    print("Estimated difference in means:", effect_est)
    Estimated difference in means: 3.1

    z = ipat.ya - ipat.yb - effect_est
    simulated_pvalues = np.zeros(1000)
    for i in range(1000):
        prng.shuffle(z)
        sim_sam = z.copy() + effect_est
        simulated_pvalues[i] = one_sample(sim_sam, stat='mean', alternative='greater', seed=1234, reps=1000)[0]
    power = np.mean(simulated_pvalues <= alpha)
    print("Estimated power:", power)
    Estimated power: 1.0

    # Job data
    effect_est = job_res[1]
    print("Estimated difference in means:", effect_est)
    Estimated difference in means: 17.29166666666667

    xnorm = job.x
    xnorm[job.y == 1] = job.x[job.y == 1] - effect_est
    simulated_pvalues = np.zeros(1000)
    for i in range(1000):
        prng.shuffle(xnorm)
        sim_sam = xnorm.copy()
        sim_sam[job.y==1] = sim_sam[job.y==1] + effect_est
        simulated_pvalues[i] = two_sample(sim_sam[job.y == 1], sim_sam[job.y == 2], stat='mean', reps = 10**3, alternative='greater', seed=1234)[0]
    power = np.mean(simulated_pvalues <= alpha)
    print("Estimated power:", power)
    Estimated power: 0.96
