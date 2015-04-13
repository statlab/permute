Kenya
=====

The Kenya dataset :cite:`1977:corrain` contains 16 observations and two variables
in total.  It concerns an anthropological study on the "Ol Molo" and "Kamba"
populations described above. Table 1 shows the sample frequencies of the 16
phenotypic combinations in the samples selected from the two populations.

Given :math:`X_1, X_2, \dots, X_n` and ...

.. ipython:: python

   from __future__ import print_function

   from permute.data import kenya

   d = kenya()

   print(d)

   import matplotlib.pyplot as plt

   plt.plot(d['ol_molo'])

   plt.plot(d['kamba'])

   plt.legend(['ol_molo', 'kamba'])

   @savefig test.png
   plt.show()

.. rubric:: References

.. bibliography:: ../../permute.bib
