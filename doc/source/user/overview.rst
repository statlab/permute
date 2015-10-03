Overview
========


Benefits of Permutation Tests
-----------------------------

-  Gives a test with correct type 1 error rate, without needing
   uncheckable assumptions
-  We can get confidence intervals for the p-value -- this tells us how
   many iterations we need to achieve a certain precision
-  Choice of test statistic is flexible -- choose one that makes sense
   for your data, gives high power

   -  Now, you can specify 'mean' or 't'
   -  Future improvements: user can choose from a library of test
      statistics or write their own function

.. ipython:: python

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import permute
    from permute import core, stratified

    from scipy import stats

Two-sample permutation tests
============================

Motivating example - teaching evaluations
-----------------------------------------

Do students give higher ratings to male teachers?

An online experiment was done with two professors, one male and one
female.

Each professor taught two sections. In one section, they used a male
name. In the other, they used a female name. The students didn't know
the teacher's real gender.

We can test whether student evaluations of teaching are biased by
comparing the ratings when the professor used a male name vs. a female
name.

The data come from MacNell et. al. (2014).

.. ipython:: python

    from permute import data
    ratings = pd.DataFrame(data.macnell2014())
    ratings.loc[:,'knowledgeable':'taidgender'].head()


How should we analyze the data?
===============================

We'll use the unpaired test statistic

.. math::  t = \frac{\text{mean(rating for M-identified prof) - mean(rating for F-identified prof)}}{\sqrt{\text{pooled SD of ratings}}}

t-test
------

Null hypothesis: reported/perceived teacher gender has no effect on
teacher ratings

Alternative hypothesis: teacher ratings differ by reported/perceived
teacher gender

Assumptions: - Ratings are normally distributed - They are on a Likert
1-5 scale. Definitely not normal - Noise is zero-mean and constant
variance across raters - How to interpret "noise" in this context? -
Constant variance is not plausible: some raters might give a range of
scores, other raters might always give 5 - Independence between
observations - Students might talk about ratings with their peers in the
class, creating dependence

.. ipython:: python

    maleid   = ratings.overall[ratings.taidgender==1]
    femaleid = ratings.overall[ratings.taidgender==0]
    df = len(maleid) + len(femaleid) - 2
    (t, p) = stats.ttest_ind(maleid, femaleid)
    print('Test statistic:', np.round(t, 5))
    print('P-value (two-sided):', np.round(p, 5))


Permutation test
----------------

Same statistic, different way to calculate p-value

 Null hypothesis: reported/perceived teacher gender has no effect
whatsoever on ratings — as if "male" and "female" are randomly assigned
labels.

 Alternative hypothesis: reported/perceived teacher gender has some
effect on ratings.

 Assumptions: - Randomization is fair and independent across units -
This can be verified from the experimental design - That's it!

.. ipython:: python

    p, t, dist = permute.core.two_sample(maleid, femaleid, stat='t', alternative='two-sided', keep_dist=True)
    print('Test statistic:', np.round(t, 5))
    print('P-value (two-sided):', np.round(p, 5))
    print('95% Confidence Interval for the P-value', np.round(ci, 5))

    print('\n\nRuns faster, but the error around the p-value is larger\n')
    p, t = permute.core.two_sample(maleid, femaleid, reps=100, stat='t', alternative='two-sided')

    print('Test statistic:', np.round(t, 5))
    print('P-value (two-sided):', np.round(p, 5))
    print('95% Confidence Interval for the P-value', np.round(ci, 5))


.. ipython:: python

    n, bins, patches = plt.hist(dist, 20, histtype='bar')
    plt.title('Permutation Null Distribution')
    plt.axvline(x = t, color = 'red')
    x = np.linspace(stats.t.ppf(0.0001, df),
                  stats.t.ppf(0.9999, df), 100)
    plt.plot(x, stats.t.pdf(x, df)*(100000/2.0), lw=2, alpha=0.6, label='t pdf')



Stratified Permutation Tests
============================

Some experimental designs have natural groupings. It makes sense to
estimate effects within groups, then combine within-group estimates.

Carry out permutation within groups, then aggregate test statistics
across groups. This helps control for group-level effects.

More on teaching evaluations
----------------------------

Do students' evaluations of teachers measure teaching effectiveness?

There are 8 professors with multiple student ratings - group the
analysis by professor

One might think that professor 1 gets consistently high reviews because
of something unrelated to teaching effectiveness, e.g. gender,
attractiveness, likeability. Stratifying by professor allows us to
assess teaching effectiveness for each individual student.

The data come from Boring et. al. (2015).

.. ipython:: python

    evals = pd.read_csv("SET2.csv")
    print(evals.head())

Use final exam performance as a proxy for value added by teacher --
students have different teachers but take the same final exam

Null hypothesis: there is no association between a student's final exam
performance and the rating they give their professor

Alternative hypothesis: there is a positive association between a
student's final exam performance and the rating they give their
professor

The test statistic we use within groups is the Spearman correlation

.. ipython:: python

    (rho, plower, pupper, pboth, sim) = permute.stratified.sim_corr(x=evals.rating, y=evals.final,
                                                                    reps=100, group=evals.prof_id)
    print('Test statistic:', np.round(rho, 5))
    print('One-sided (upper) p-value:', np.round(pupper, 5))

    n, bins, patches = plt.hist(sim, 20, histtype='bar')
    plt.title('Permutation Null Distribution')
          plt.axvline(x = rho, color = 'red')

    @savefig
    plt.show()

References
----------

Boring, A., Ottoboni, K., and Stark, P.B. (in preparation), "Student
Evaluations of Teaching (Mostly) Do Not Measure Teaching Effectiveness."

MacNell, L., Driscoll, A., and Hunt, A.N. (2014), "What’s in a Name:
Exposing Gender Bias in Student Ratings of Teaching," Innovative Higher
Education, 1-13.

----

.. automodule:: permute
