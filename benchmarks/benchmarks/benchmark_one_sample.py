import numpy as np

from permute.core import one_sample


class OneSample:

    def setup(self):
        self.prng = np.random.RandomState(42)
        # short example
        self.t = [689, 656, 668, 660, 679, 663, 664, 647, 694, 633, 653]
        self.c = [657, 623, 652, 654, 658, 646, 600, 640, 605, 635, 642]
        self.d = np.array(self.t) - np.array(self.c)
        # longer example
        self.t_long = self.t * 10
        self.c_long = self.c * 10
        self.d_long = np.array(self.t_long) - np.array(self.c_long)


    def time_one_sample(self):
        p, diffs, dist = one_sample(self.d, stat='t', keep_dist=True, seed=self.prng)


    def time_one_sample_large(self):
        p, diffs, dist = one_sample(self.d, stat='t', keep_dist=True, seed=self.prng)

