User Guide
==========

Permutation tests (sometimes referred to as randomization, re-randomization, or
exact tests) are a nonparametric approach to statistical significance testing.
They were first introduced by R. A. Fisher in 1935 :cite:`fisher1935design` and
further developed by E. J. G. Pitman
:cite:`pitman1937,pitman1938significance`.  After the introduction of the
bootstrap, the ideas were extended in the 1980's by J. Romano
:cite:`romano1988bootstrap,romano1989bootstrap`.

Permutation tests were developed to test hypotheses for which relabeling the
observed data was justified by
exchangeability [#f1]_
of the observed random variables.  In these situations, the
conditional distribution of the test statistic under the null hypothesis is completely
determined by the fact that all relabelings of the data are equally likely.
That distribution might be calculable in closed form; if not, it can be simulated
with arbitrary accuracy by generating relabelings uniformly at random.
In contrast to approximate parametric methods or asymptotic methods, the accuracy
of the simulation for any finite (re)sample size is known, and can be made
arbitrarily small at the expense of computing time.

More generally, permutation tests are possible whenever the null
distribution of the data is invariant under the action of some group
(see Appendix [app:def] for background). Then, a subset of outcomes is
conditionally equally likely, given that the data fall in a particular
*orbit* of the group (all potential observations that result from
applying elements of the group to the observed value of the data). That
makes it possible to determine the conditional distribution of any test
statistic, given the orbit of the data. Since the conditional
distribution is uniform on the orbit of the original data, the
probability of any event is the proportion of possible outcomes that lie
in the event. If tests are performed conditionally at level
:math:`\alpha` regardless of the observed data, the resulting overall
test has unconditional level :math:`\alpha`, by the law of total
probability.

.. toctree::
   :maxdepth: 2

   one-sample.rst
   two-sample.rst
   regression.rst
   npc/index.rst
   references.rst

.. rubric:: Footnotes


.. [#f1] A sequence $X_1, X_2, X_3, \dots, X_n$ of random variables is
   *exchangeable* if their joint distribution is invariant to
   permutations of the indices; that is,

   .. math::
              p(x_1, \dots, x_n) = p(x_{\pi(1)}, \dots, x_{\pi(n)})

   for all permutations $\pi$ of $\{1, 2, \dots, n\}$.  It is closely related to the
   notion of *independent and identically-distributed* random variables.
   Independent and identically-distributed random variables are exchangeable.
   However, simple random sampling *without* replacement produces an
   exchangeable, but not independent, sequence of random variables.
