Regression
----------

Given :math:`n=10` observations of two scalars :math:`(x_i, y_i)` for
:math:`i = 1, 2, \dots, n`, consider the simple linear regression model

.. math::

  y_i = a + bx_i + \epsilon_i.

Assume that :math:`\{\epsilon_i\}_{i=1}^{10}` are exchangeable.

You are interested in testing whether the slope of the population
regression line is non-zero; hence, your null hypothesis is
:math:`b = 0`. If :math:`b = 0`, then the model reduces to
:math:`y_i = a + \epsilon_i` for all :math:`i`. If this is true, the
:math:`\{y_i\}_{i=1}^{10}` are exchangeable since they are just shifted
versions of the exchangeable :math:`\{\epsilon_i\}_{i=1}^{10}`. Thus
every permutation of the :math:`\{y_i\}_{i=1}^{10}` has the same
conditional probability regardless of the :math:`x`\ s. Hence every
pairing :math:`(x_i, y_j)` for any fixed :math:`i` and for
:math:`j = 1, 2, \dots, n` is equally likely.

Using the least squares estimate of the slope normalized by its standard
error as the test statistic, you can find its exact distribution under
the null given the observed data by computing the test statistic on all
possible pairs formed by permuting the :math:`y` values, keeping the
original order of the :math:`x` values. From the distribution of the
test statistic under the null conditioned on the observed data, the is
the ratio of the count of the *as extreme* or *more extreme* test
statistics to the total number of such test statistics. For :math:`n=10`
you might in principle enumerate all :math:`10!` equally likely pairings
and then compute the exact . For sufficiently large :math:`n`,
enumeration becomes infeasible; in which case, you could approximate the
exact using a uniform random sample of the equally likely pairings.

A parametric approach to this problem would begin by imposing additional
assumptions on the noise :math:`\epsilon`. For example, if we assume
that :math:`\{\epsilon_i\}` are iid Gaussians with mean zero, then the
test statistic has a :math:`t`-distribution with :math:`n-2` degrees of
freedom. If this additional assumption holds, then we can read the off a
table. Note that, unlike in the permutation test, we were only able to
calculate the (even with the additional assumptions) because we happened
to be able to derive the distribution of this specific test statistic.
