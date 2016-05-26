from numpy import array, mean, sqrt
from numpy.random import RandomState
from permute.core import one_sample


def time_one_sample():
    t = [689, 656, 668, 660, 679, 663, 664, 647, 694, 633, 653]
    c = [657, 623, 652, 654, 658, 646, 600, 640, 605, 635, 642]
    d = array(t) - array(c)
    n = len(d)

    prng = RandomState(42)
    p, diff_means, dist = one_sample(d, stat='t', keep_dist=True, seed=prng)


def time_one_sample_large():
    t = [689, 656, 668, 660, 679, 663, 664, 647, 694, 633, 653] * 10
    c = [657, 623, 652, 654, 658, 646, 600, 640, 605, 635, 642] * 10
    d = array(t) - array(c)
    n = len(d)

    prng = RandomState(42)
    p, diff_means, dist = one_sample(d, stat='t', keep_dist=True, seed=prng)
