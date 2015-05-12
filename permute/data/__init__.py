"""Standard test data.

For more information, see

 - http://www.wiley.com/legacy/wileychi/pesarin/material.html

"""

import os as _os

import numpy as np

from .. import data_dir


__all__ = ['load',
           'kenya', ]


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
    return np.recfromcsv(_os.path.join(data_dir, f), delimiter=",")


def nsgk():
    """NSGK test data for irr.
    """
    nz = np.loadtxt(_os.path.join(data_dir, "nsgk.csv"),
                    delimiter=',', skiprows=1, dtype=np.int)
    shape = tuple(nz.max(axis=0))
    x = np.zeros(shape, dtype=np.int)
    nz -= 1
    for r in nz:
        x[tuple(r)] = 1

    # given order: time_stamp,domain,video,rater
    # desired order: domain,video,rater,time_stamp
    x = x.transpose(1, 2, 3, 0)
    # hardcoding the number of timestamps per video
    time_stamps = [36, 32, 35, 37, 31, 35, 40, 32]
    p1 = [[m[:, :time_stamps[i]] for i, m in enumerate(n)]for n in x]

    ## Alternatively, I could return a 2D object array with
    ##  rater x time_stamp(video) matrices as entries
    ## Not sure which is better, so I will wait to see how I use it.
    # p1 = np.zeros(x.shape[:2], dtype=object)
    # for i, n in enumerate(x):
    #     for j, m in enumerate(n):
    #        p1[i, j] = m

    return p1


def macnell2014():
    """Data from MacNell et alrom MacNell et alrom MacNell et al. 2014

    .. Lillian MacNell, Adam Driscoll, and Andrea N Hunt, "What's
       in a Name: Exposing Gender Bias in Student Ratings of Teaching,"
       Innovative Higher Education, pp. 1-13, 2014.
    """
    return load("MacNell2014.csv")


# def another_poss():
#    nz = np.loadtxt(_os.path.join(data_dir, "nsgk.csv"),
#                    delimiter=',', skiprows=1, dtype=np.int)
#    _, nd, nv, nr = tuple(nz.max(axis=0))
#    dv = np.zeros((nd, nv), dtype=object)
#    time_stamps = [36, 32, 35, 37, 31, 35, 40, 32]
#    for n in range(nd):
#        for v in range(nv):
#            dv[n, v] = np.zeros((nr, time_stamps[v]), dtype=np.int)
#    nz -= 1
#    for _ts, _d, _v, _r in nz:
#        dv[_d, _v][_r, _ts] = 1 
#

def botulinum():
    """The

    """
    return load(_os.path.join("npc", "botulinum.csv"))


def chrom17m():
    """The

    """
    return load(_os.path.join("npc", "chrom17m.csv"))


def confocal():
    """The

    """
    return load(_os.path.join("npc", "confocal.csv"))


def germina():
    """The

    """
    return load(_os.path.join("npc", "germina.csv"))


def kenya():
    """The Kenya dataset contains 16 observations and two variables in total.
    It concerns an anthropological study on the "Ol Molo" and "Kamba"
    populations.

    """
    return load(_os.path.join("npc", "kenya.csv"))


def massaro_blair():
    """The

    """
    return load(_os.path.join("npc", "massaro_blair.csv"))


def monachus():
    """The

    """
    return load(_os.path.join("npc", "monachus.csv"))


def mult():
    """The

    """
    return load(_os.path.join("npc", "mult.csv"))


def perch():
    """The

    """
    return load(_os.path.join("npc", "perch.csv"))


def rats():
    """The

    """
    return load(_os.path.join("npc", "rats.csv"))


def setig():
    """The

    """
    return load(_os.path.join("npc", "setig.csv"))


def urology():
    """The

    """
    return load(_os.path.join("npc", "urology.csv"))


def washing_test():
    """The

    """
    return load(_os.path.join("npc", "washing_test.csv"))


def waterfalls():
    """The

    """
    return load(_os.path.join("npc", "waterfalls.csv"))
