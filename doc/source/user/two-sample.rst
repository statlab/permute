Two sample permutation tests
============================

To illustrate the two-sample permutation test, consider the following
randomized, controlled experiment. You suspect a specific treatment will
increase the growth rate of a certain type of cell. To test this hypothesis,
you clone 100 cells. Now there are 200 cells composed of 100 pairs of identical
clones. For each cloned pair you randomly assign one to treatment, with
probability 1/2, independently across the 100 pairs. At the end of the
treatment, you measure the growth rate for all the cells. The null hypothesis
is that treatment has no effect. If that is true, then the assignment of a
clone to treatment amounts to an arbitrary label that has nothing to do with
the measured response. So, given the responses within each pair (but not the
knowledge of which clone in each pair had which response), it would have been
just as likely to observe the same *numbers* but with flipped labels within
each pair. We could generate new hypothetical datasets from the observed data
by assigning the treatment and control labels for all the cloned pairs
independently.  This yields a total of :math:`2^{100}` total datasets
(including the observed data and all the hypothetical datasets that you
generated), all equally likely to have occurred under the null, conditioning on
the observed data (but not the labeling).

The standard parametric approach to this problem is the paired :math:`t`-test,
since the cloned cells are presumably more similar to each other than to
another randomly chosen cell (and thus more readily compared). The paired
:math:`t`-test assumes that, if the null hypothesis is true, the differences in
response between each pair of clones are independently and identically (iid)
normally distributed with mean zero and unknown variance. The test statistic is
the mean of the differences between each cloned pair divided by the standard
error of these differences. Under these assumptions, the test statistic is
distributed as a :math:`t`-distribution with :math:`n-1` degrees of freedom.
This means you can calculate the test statistic and then read off the from the
:math:`t`-distribution. If the is below some prespecified critical value
:math:`\alpha`, then you reject the null. If the true generative model for the
data is not iid normal, however, the probability of rejecting the null
hypothesis can be quite different from :math:`\alpha` even if treatment has no
effect.

A permutation version of the :math:`t`-test can avoid that vulnerability: one
can use the :math:`t`-statistic as the test statistic, but instead of selecting
the critical value on the basis of Student’s :math:`t`-distribution, one uses
the distribution of the statistic under the permutation distribution. Of
course, other test statistics could be used instead; the test statistic should
be sensitive to the nature of the alternative hypothesis, to ensure that the
test has power against the alternatives the science suggests are relevant.

Regardless of which test statistic you choose for your permutation test, if the
problem size is not too large then you enumerate all equally likely
possibilities under the null given the observed data. If the problem is too
large to feasibly enumerate, then you use a suitably large, iid random sample
from the exact distribution just described, by selecting permutations uniformly
at random and applying the test statistic to those permutations. As you
increase the number of samples, you will get increasingly better (in
probability) approximations of the exact distribution of the test statistic
under the null. The null conditional probability of any event can be estimated
as the proportion of random permutations for which the event occurs, and the
sampling variability of that estimate can be characterized exactly, for
instance, using binomial tests (since the distribution of the number of times
the event occurs is Binomial with :math:`n` equal to the number of samples and
:math:`p` the unknown probability to be estimated).

Gender bias in student evaluation of teachers
---------------------------------------------

There is growing evidence of gender bias in student evaluations of
teaching. To address the question “Do students give higher ratings to
male teachers?,” an online experiment was done with two professors, one
male and one female \cite{macnell2014s}. Each professor taught two sections. In one
section, they used a male name. In the other, they used a female name.
The students didn’t know the teacher’s real gender. We test whether
student evaluations of teaching are biased by comparing the ratings when
the professor used a male name versus a female name.

Parametric Approach
~~~~~~~~~~~~~~~~~~~

First let us consider the parametric two-sample t-test. In this case, our test
statistic is

.. math::

   t = \frac{\text{mean(rating for M-identified prof) - mean(rating for F-identified prof)}}{\sqrt{\text{pooled SD of ratings}}}

For the two-sample t-test, the null hypothesis is that the reported/perceived
teacher gender has no effect on teacher ratings. The alternative hypothesis is
that teacher ratings differ by reported/perceived teacher gender. For the
two-sample t-test to be valid, we require the following assumptions:

-  Ratings are normally distributed. (But they are on a Likert 1-5
   scale, which is definitely not normal.)

-  Noise is zero-mean and constant variance across raters. (How should
   we interpret “noise” in this context? Besides constant variance is
   not plausible: some raters might give a range of scores, other raters
   might always give 5.)

-  Independence between observations. (Students might talk about ratings
   with their peers in the class, creating dependence.)

Despite the problematic assumptions we are required to make, let’s temporarily
assume they hold and calculate a “:math:`p`-value” anyway.

.. plot::
    :context:
    :nofigs:

    >>> from __future__ import print_function
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats

    >>> from permute.data import macnell2014
    >>> ratings = macnell2014()
    >>> maleid = ratings.overall[ratings.taidgender==1]
    >>> femaleid = ratings.overall[ratings.taidgender==0]
    >>> df = len(maleid) + len(femaleid) - 2
    >>> t, p = stats.ttest_ind(maleid, femaleid)
    >>> print('Test statistic:', np.round(t, 5))
    Test statistic: 1.54059
    >>> print('P-value (two-sided):', np.round(p, 5))
    P-value (two-sided): 0.1311

Note that the computed “:math:`p`-value” is above the standard cut-offs for
reporting significance in the literature.

Permutation approach
~~~~~~~~~~~~~~~~~~~~

For the permutation test we can use the same test statistic, but we will
compute the :math:`p`-value by randomly sampling the exact distribution of the
test statistics. The null hypothesis is that the ratings are uninfluenced by
instructor---any particular student would assign the same rating to either
instructor.  The alternative hypothesis is that the ratings differ by
instructor---some students would assign different ratings to the two
instructors.  The only assumption we need to make is that randomization is fair
and independent across units. This can be verified directly from the
experimental design.

.. plot::
    :context:
    :nofigs:

    >>> from permute.core import two_sample
    >>> p, t = two_sample(maleid, femaleid, stat='t', alternative='two-sided')
    >>> print('Test statistic:', np.round(t, 5))
    Test statistic: 1.82159
    >>> print('P-value (two-sided):', np.round(p, 5))
    P-value (two-sided): 0.04436

    >>> print('\n\nRuns faster, but there is more uncertainty around the p-value\n')
    >>> p, t = two_sample(maleid, femaleid, reps=100, stat='t', alternative='two-sided')
    >>> print('Test statistic:', np.round(t, 5))
    >>> print('P-value (two-sided):', np.round(p, 5))

Not only is the use of this :math:`p`-value justified (since our assumptions
are met), but its value is below the cut-off for significance commonly used.
Since the permutation test also returns the approximately exact distribution of
the test statistic, let’s compare the actual distribution with the
:math:`t`-distribution.

.. plot::
    :context:

    >>> p, t, distr = two_sample(maleid, femaleid, stat='t', reps=10000, 
    ...                          alternative='greater', keep_dist=True, seed=55)
    >>> n, bins, patches = plt.hist(distr, 25, histtype='bar', normed=True)
    >>> plt.title('Permutation Null Distribution')
    >>> plt.axvline(x = -t, color = 'red')
    >>> x = np.linspace(stats.t.ppf(0.0001, df),
    ...       stats.t.ppf(0.9999, df), 100)
    >>> plt.plot(x, stats.t.pdf(x, df), lw=2, alpha=0.6)

The plot above shows the null distribution generated by 10,000 permutations of the data.
The t distribution is superimposed for comparison.  The null distribution is much flatter
around 0 than the t distribution.  This is the source of the difference in p-values between
the two tests.


Stratified Spearman correlation permutation test
------------------------------------------------

Some experimental designs have natural groupings. It makes sense to estimate
effects within groups, then combine within-group estimates.

To turn this idea into a permutation test, we carry out permutations within
groups, then aggregate the test statistics across groups. This helps control
for group-level effects.

More on teaching evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We established that one instructor got higher ratings from students, but the
difference was not significant. Now we may ask, did students ratings differ
according to the gender that the instructor claimed to be?

If there is no gender bias in the ratings, then students should give the same
rating to the male instructor regardless of the gender he claims to be and
students should give the same rating to the female instructor regardless of the
gender she claims to be. However, we don't necessarily believe that students
would rate the two instructors the same, since there may be some difference in
their teaching styles.

Null hypothesis: student by student, the instructor would receive the same
rating regardless of reported gender

Alternative hypothesis: there is at least one student who would rate their
instructor higher if they identified as male

The test statistic we use within groups is the Spearman correlation. For each
instructor, we compute the correlation between their rating and reported
gender, then add the absolute values of the correlations for the instructors.

.. plot::
    :context:

    >>> from permute.stratified import sim_corr
    >>> evals = np.recfromcsv("SET2.csv")
    >>> rho, plower, pupper, pboth, sim = sim_corr(x=evals.rating, y=evals.final,
    ...                                            group=evals.prof_id)
    >>> print 'Test statistic:', np.round(rho, 5)
    Test statistic: 0.94787
    >>> print 'One-sided (upper) P-value:', np.round(pupper, 5)
    One-sided (upper) P-value: 0.18

Finally, I plot the simulated distribution of the test statistics under
the null conditioned on the observed data in Figure [fig:figure2].

.. plot::
    :context:

    >>> n, bins, patches = plt.hist(sim, 40, histtype='bar')
    >>> plt.axvline(x=rho, color='red')
    >>> plt.show()

At the 10% level, there is a significant difference in ratings between
male-identified and female-identified instructors. We could not have computed
this p-value with any common distribution, since the null hypothesis assumes
some observations (ratings for a single instructor) are exchangeable but others
are not.
