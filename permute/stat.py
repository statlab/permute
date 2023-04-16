
"""
Some common test statistics for functions in core.py or ksample.py
"""
from scipy.stats import rankdata
import numpy as np
def wilcoxon(x,y):
    r"""
    calculate the Wilcoxon rank-sum test statistic 

    Parameters
    ----------
    x : array-like
        Sample 1
    y : array-like
        Sample 2

    Returns
    -------
    float
        the Wilcoxon rank-sum test statistic
        $W_X=\sum_j Z_jR_j$ where $R_j$ is the (mid) rank and $Z_j$ denotes whether subject j is in Sample 1
    """
    x = np.array(x)
    y = np.array(y)
    midrank=rankdata(np.concatenate((x,y)))
    return np.sum(midrank[:len(x)])

def smirnov(x,y):
    r"""
    calculate the Kolmogorov-Smirnov test statistic 

    Parameters
    ----------
    x : array-like
        Sample 1
    y : array-like
        Sample 2

    Returns
    -------
    float
        the Kolmogorov-Smirnov test statistic
        $\max_t |F_x(t) - F_y(t)|$ where $F$ is the empirical CDF.
    """
    return np.max( \
                [abs(sum(x<=val)/len(x)-sum(y<=val)/len(y)) for val in np.concatenate([x, y])]\
                )

def siegel_tukey(x,y):
    r"""
    calculate the Siegel-Tukey test statistic for
    testing for a difference in dispersion

    Note if apply this test statistic to function two_sample,
    the alternative will be

    (a) 'greater' if the alternative is Sample 1 has less dispersion
    (b) 'less' if the alternative is Sample 1 has larger dispersion
    (b) 'two-sided' if the alternative is Sample 1 and Sample 2 have different dispersion 

    Parameters
    ----------
    x : array-like
        Sample 1
    y : array-like
        Sample 2

    Returns
    -------
    float
        the Siegel-Tukey test statistic
        $W_X=\sum_j Z_jR_j$ where $R_j$ is the (mid) rank and $Z_j$ denotes whether subject j is in Sample 1.

        Here the rank is assigned differently: We assign rank 1 to the lowest number of the sequence, 
        ranks 2 to the highest numbers in the sequence,
        ranks 3 to the next lowest, etc
    """
    x = np.array(x)
    y = np.array(y)
    #Sort the data
    data=[['x',x[i]] for i in range(len(x))]+[['y',y[i]] for i in range(len(y))]
    sortdata=sorted(data, key=lambda d: d[1], reverse=False)
    # Calculate the rank to measure the dispersion
    stat=np.zeros(len(sortdata))
    head=0
    tail=len(sortdata)-1
    rank=0
    while head<=tail:
        temp_head=head
        temp_tail=tail
        head+=1
        rank+=1
        while sortdata[head][1]==sortdata[temp_head][1]:
            head+=1
            rank+=1
        stat[temp_head:head]=rank-(head-temp_head-1)/2
        if head>tail:
            break
        tail-=1
        rank+=1
        while sortdata[tail][1]==sortdata[temp_tail][1]:
            tail-=1
            rank+=1
        stat[tail+1:temp_tail+1]=rank-(temp_tail-tail-1)/2
    return np.sum([stat[i] for i in range(len(sortdata)) if sortdata[i][0]=='x'])

def wilcoxon_signed(x):
    r"""
    calculate the Wilcoxon signed rank-sum test statistic 

    Parameters
    ----------
    x : array-like
        Sample 

    Returns
    -------
    float
        the Wilcoxon rank-sum test statistic
        $W_X=sign(x_j)R_j$ where $R_j$ is the (mid) rank of the absolute value of the observed data
    """
    x = np.array(x)
    midrank=rankdata(abs(np.array(x)))
    return np.sum(np.sign(x)*midrank)