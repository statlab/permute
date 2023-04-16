import numpy as np

from ..stat import (wilcoxon,
                    wilcoxon_signed,
                    siegel_tukey,
                    smirnov
                   )
x=np.array([1,2,2,5])
y=np.array([3,3,3,4])
z=x-y
assert wilcoxon(x,y)==14
assert wilcoxon_signed(z)==-6
assert siegel_tukey(x,y)==10
assert smirnov(x,y)==0.75