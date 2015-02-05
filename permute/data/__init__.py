"""Standard test data.

For more images, see

 - http://www.wiley.com/legacy/wileychi/pesarin/material.html

"""

import os as _os

import pandas as pd

from .. import data_dir


__all__ = ['load',
           'kenya']

def load(f):
    """Load a data file located in the data directory.

    Parameters
    ----------
    f : string
        File name.

    Returns
    -------
    x : ndarray (or Pandas' frame?)
        Data loaded from permute.data_dir.
    """
    return pd.read_csv(_os.path.join(data_dir, f))

def kenya():
    """The Kenya dataset contains 16 observations and two variables in total.
    It concerns an anthropological study on the "Ol Molo" and "Kamba"
    populations

    """
    return load("kenya.csv")
