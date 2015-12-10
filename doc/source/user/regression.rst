Regression
----------

Given :math:`n` observations of two scalars :math:`(x_i, y_i)` for
:math:`i = 1, 2, \dots, n`, consider the simple linear regression model

.. math::

  y_i = a + bx_i + \epsilon_i.

Assume that :math:`\{\epsilon_i\}_{i=1}^{n}` are exchangeable.

You are interested in testing whether the slope of the population
regression line is non-zero; hence, your null hypothesis is
:math:`b = 0`. If :math:`b = 0`, then the model reduces to
:math:`y_i = a + \epsilon_i` for all :math:`i`. If this is true, the
:math:`\{y_i\}_{i=1}^{n}` are exchangeable since they are just shifted
versions of the exchangeable :math:`\{\epsilon_i\}_{i=1}^{n}`. Thus
every permutation of the :math:`\{y_i\}_{i=1}^{n}` has the same
conditional probability regardless of the :math:`x`\ s. Hence every
pairing :math:`(x_i, y_j)` for any fixed :math:`i` and for
:math:`j = 1, 2, \dots, n` is equally likely.

Using the least squares estimate of the slope as the test statistic, you can
find its exact distribution under the null given the observed data by computing
the test statistic on all possible pairs formed by permuting the :math:`y`
values, keeping the original order of the :math:`x` values. From the
distribution of the test statistic under the null conditioned on the observed
data, the is the ratio of the count of the *as extreme* or *more extreme* test
statistics to the total number of such test statistics. You might in principle
enumerate all :math:`n!` equally likely pairings and then compute the exact p-value.
For sufficiently large :math:`n`, enumeration becomes infeasible; in which
case, you could approximate the exact p-value using a uniform random sample of the
equally likely pairings.

A parametric approach to this problem would begin by imposing additional
assumptions on the noise :math:`\epsilon`. For example, if we assume
that :math:`\{\epsilon_i\}` are iid Gaussians with mean zero, then the
the least squares estimate of the slope normalized by its standard
error has a :math:`t`-distribution with :math:`n-2` degrees of
freedom. If this additional assumption holds, then we can read the off a
table. Note that, unlike in the permutation test, we were only able to
calculate the p-value (even with the additional assumptions) because we happened
to be able to derive the distribution of this specific test statistic.

Derivation
~~~~~~~~~~

Given $n$ observations

.. math::

  y_i = a + bx_i + \epsilon_i,

the least square solution is

.. math::

  \min_{a, b} \sum_{i=1}^n \left(y_i - a - bx_i \right)^2


Taking the partial derivative with respect to $a$

.. math::

  \frac{\partial }{\partial b} \sum_{i=1}^n \left(y_i - a - bx_i \right)^2
   &= -2 \sum_{i=1}^n \left(y_i - a - bx_i \right) \\
   &= -2 \left(\sum_{i=1}^n y_i - na - b \sum_{i=1}^n x_i \right) \\
   &= -2n \left( \bar{y} - a - b \bar{x} \right).

Setting this to $0$ and solving for $a$ yields our estimate $\hat{a}$

.. math::

  \hat{a} = \bar{y} - b \bar{x}.

Taking the partial derivative with respect to $b$

.. math::

  \frac{\partial }{\partial b} \sum_{i=1}^n \left(y_i - a - bx_i \right)^2
   &= -2 \sum_{i=1}^n \left(y_i - a - bx_i \right) x_i \\
   &= -2 \left(\sum_{i=1}^n y_ix_i - a \sum_{i=1}^n x_i - b \sum_{i=1}^n x_ix_i \right) \\
   &= -2n \left( \overline{xy} - a \bar{x} - b \overline{xx} \right).

Plugging in $\hat{a}$, setting the result to $0$, and solving for $b$ yields

.. math::

  \hat{b} &= \frac{\overline{xy} - \bar{x}\bar{y}}{\overline{xx} - \bar{x}\bar{x}}
    = \frac{\mathrm{Cov}(x, y)}{\mathrm{Var}(x)} = \mathrm{Cor}(x, y)\left(\frac{\mathrm{Std}(y)}{\mathrm{Std}(x)}\right).

Since $\frac{\mathrm{Std}(y)}{\mathrm{Std}(x)}$ is constant under the
permutation of $y$, we can calculate the p-value using the permutation
test of the correlation.

.. code::

    >>> from __future__ import print_function
    >>> import numpy as np

    >>> X = np.array([np.ones(10), np.random.random_integers(1, 4, 10)]).T
    >>> beta = np.array([1.2, 2])
    >>> epsilon = np.random.normal(0, .15, 10)
    >>> y = X.dot(beta) + epsilon

    >>> from permute.core import corr
    >>> t, pv_left, pv_right, pv_both, dist = corr(X[:, 1], y)
    >>> print(t)
    0.998692462616
    >>> print(pv_both)
    0.0007
    >>> print(pv_right)
    0.0007
    >>> print(pv_left)
    1.0

    >>> t, pv_both, dist = corr(X[:, 1], y)
    >>> print(t)
    0.103891027265
    >>> print(pv_both)
    0.765
    >>> print(pv_right)
    0.3818
    >>> print(pv_left)
    0.619
