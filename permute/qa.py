""" Quality assurance and data cleaning.
"""

from __future__ import division, print_function, absolute_import

import numpy as np


def find_duplicate_rows(x, as_string=False):
    """ Find rows which are duplicated in x

    Notes
    -----
    If you load a file, for example `nsgk.csv`, as a 2D array, say `x`,
    then if you found '16,20,2,8' in the list returned by
    ``find_duplicate_rows(x, as_string=True)`` you might do something like:

        $ grep -n --context=1 '16,20,2,8' nsgk.csv
        12512-16,15,2,8
        12513:16,20,2,8
        12514-16,45,2,8
        --
        12532-17,17,2,8
        12533:16,20,2,8
        12534-17,24,2,8

    http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
    """
    indx = np.lexsort(x.T)
    x = x[indx]
    diff = np.diff(x, axis=0)
    indx = np.any(diff, axis=1)
    dups = x[1:, :][~indx, ]
    if as_string:
        dups = [",".join([str(c) for c in r.tolist()]) for r in dups]
    return dups


def find_consecutive_duplicate_rows(x, as_string=False):
    """ Find rows which are duplicated in x
    """
    indx = []
    prev = x[0]
    for i, r in enumerate(x[1:]):
        if (r == prev).all():
            indx.append(i)
        prev = r
    dups = x[indx]
    if as_string:
        dups = [",".join([str(c) for c in r.tolist()]) for r in dups]
    return dups
