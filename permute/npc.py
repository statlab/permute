import numpy as np
import copy
from scipy.stats import norm, rankdata, ttest_ind, ttest_1samp
from cryptorandom.sample import random_sample
from .utils import get_prng, permute


# Combining functions

def fisher(pvalues):
    r"""
    Apply Fisher's combining function

    .. math:: -2 \sum_i \log(p_i)

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine

    Returns
    -------
    float
        Fisher's combined test statistic
    """
    return -2 * np.log(np.prod(pvalues))


def liptak(pvalues):
    r"""
    Apply Liptak's combining function

    .. math:: \sum_i \Phi^{-1}(1-p_i)

    where $\Phi^{-1}$ is the inverse CDF of the standard normal distribution.

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine

    Returns
    -------
    float
        Liptak's combined test statistic
    """
    return np.sum(norm.ppf(1 - pvalues))


def tippett(pvalues):
    r"""
    Apply Tippett's combining function

    .. math:: \max_i \{1-p_i\}

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine

    Returns
    -------
    float
        Tippett's combined test statistic
    """
    return np.max(1 - pvalues)


def inverse_n_weight(pvalues, size):
    r"""
    Compute the test statistic

    .. math:: -\sum_{s=1}^S \frac{p_s}{\sqrt{N_s}}

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine
    size : array_like
        The $i$th entry is the sample size used for the $i$th test

    Returns
    -------
    float
        combined test statistic
    """
    weights = size ** (-1 / 2)
    return np.sum(-1 * pvalues * weights)


# Nonparametric combination of tests

def check_combfunc_monotonic(pvalues, combfunc):
    r"""
    Utility function to check that the combining function is monotonically
    decreasing in each argument.

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine
    combine : function
        The combining function to use.

    Returns
    -------
    ``True`` if the combining function passed the check, ``False`` otherwise.
    """

    obs_ts = combfunc(pvalues)
    for i in range(len(pvalues)):
        test_pvalues = pvalues.copy()
        test_pvalues[i] = test_pvalues[i] + 0.1
        if(obs_ts < combfunc(test_pvalues)):
            return False
    return True


def npc(pvalues, distr, combine="fisher", plus1=True):
    r"""
    Combines p-values from individual partial test hypotheses $H_{0i}$ against
    $H_{1i}$, $i=1,\dots,n$ to test the global null hypothesis

    .. math:: \cap_{i=1}^n H_{0i}

    against the alternative

    .. math:: \cup_{i=1}^n H_{1i}

    using an omnibus test statistic.

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine
    distr : array_like
        Array of dimension [B, n] where B is the number of permutations and n is
        the number of partial hypothesis tests. The $i$th column of distr contains
        the simulated null distribution of the $i$th test statistic under $H_{0i}$.
    combine : {'fisher', 'liptak', 'tippett'} or function
        The combining function to use. Default is "fisher".
        Valid combining functions must take in p-values as their argument and be
        monotonically decreasing in each p-value.
    plus1 : bool
        flag for whether to add 1 to the numerator and denominator of the
        p-value based on the empirical permutation distribution. 
        Default is True.

    Returns
    -------
    float
        A single p-value for the global test
    """
    n = len(pvalues)
    B = distr.shape[0]
    if n < 2:
        raise ValueError("One p-value: nothing to combine!")
    if n != distr.shape[1]:
        raise ValueError("Mismatch in number of p-values and size of distr")

    combine_library = {
        "fisher": fisher,
        "liptak": liptak,
        "tippett": tippett
    }
    if callable(combine):
        if not check_combfunc_monotonic(pvalues, combine):
            raise ValueError(
                "Bad combining function: must be monotonically decreasing in each p-value")
        combine_func = combine
    else:
        combine_func = combine_library[combine]

    # Convert test statistic distribution to p-values
    combined_stat_distr = [0] * B
    pvalues_from_distr = np.zeros((B, n))
    for j in range(n):
        pvalues_from_distr[:, j] = 1 - rankdata(distr[:, j], method="min")/(plus1+B) + (1 + plus1)/(plus1+B)
    if combine == "liptak":
        toobig = np.where(pvalues_from_distr >= 1)
        pvalues_from_distr[toobig] = 1 - np.finfo(float).eps
    combined_stat_distr = np.apply_along_axis(
        combine_func, 1, pvalues_from_distr)

    observed_combined_stat = combine_func(pvalues)
    return (plus1 + np.sum(combined_stat_distr >= observed_combined_stat)) / (plus1+B)


def sim_npc(data, test, combine="fisher", in_place=False, reps=int(10**4), seed=None):
    r''' 
    Combines p-values from individual partial test hypotheses $H_{0i}$ against
    $H_{1i}$, $i=1,\dots,n$ to test the global null hypothesis

    .. math:: \cap_{i=1}^n H_{0i}

    against the alternative

    .. math:: \cup_{i=1}^n H_{1i}

    using an omnibus test statistic.
    
    Parameters
    ----------
    data : Experiment object
    test : array_like
        Array of functions to compute test statistic to apply to each column in cols
    combine : {'fisher', 'liptak', 'tippett'} or function
        The combining function to use. Default is "fisher".
        Valid combining functions must take in p-values as their argument and be
        monotonically decreasing in each p-value.
    in_place : Boolean
        whether randomize group in place, default False
    reps : int
        number of repetitions
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator
    
    Returns
    -------
    array
        A single p-value for the global test, 
        test statistic values on the original data,
        partial p-values
    '''
    # check data is of type Experiment
    if not isinstance(data, Experiment):
        raise ValueError("data not of class Experiment")
        
    # if seed not none, reset seed
    if seed is not None:
        data.randomizer.reset_seed(seed)
    
    ts = {}
    tv = {}
    ps = {}

    # get the test statistic for each column on the original data
    for c in range(len(test)):
        # apply test statistic function to column
        ts[c] = test[c](data)
        tv[c] = []
        
    # check if randomization in place
    if in_place:
        data_copy = data
    else:
        data_copy = copy.deepcopy(data)
        
    # get test statistics for random samples
    for i in range(reps):
        # randomly permute group
        data_copy.randomize()
        # calculate test statistics on permuted data
        for c in range(len(test)):
            # get test statistic for this permutation
            tv[c].append(test[c](data_copy))
    # get p-values for original data
    for c in range(len(test)):
        ps[c] = (np.sum(np.array(tv[c]) >= ts[c]) + 1)/(reps + 1)
    # change format of dist to array
    dist = np.array([tv[c] for c in range(len(test))]).T
    # append test statistic from orignal data to dist
    dist = np.append(dist, np.array([ts[c] for c in range(len(test))], ndmin=2), axis=0)
    # run npc
    p = npc(np.array([ps[c] for c in range(len(test))]), dist, combine=combine, plus1=False)
    return p, ts, ps


def fwer_minp(pvalues, distr, combine='fisher', plus1=True):
    """
    Adjust p-values using the permutation "minP" variant of Holm's step-up method.
    
    When considering a closed testing procedure, the adjusted p-value 
    $p_i$ for a given hypothesis $H_i$ is the maximum of all p-values for tests 
    including $H_i$ as a special case (including the p-value for the $H_i$ 
    test itself).
    
    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine
    distr : array_like
        Array of dimension [B, n] where B is the number of permutations and n is
        the number of partial hypothesis tests. The $i$th column of distr contains
        the simulated null distribution of the $i$th test statistic under $H_{0i}$.
    combine : {'fisher', 'liptak', 'tippett'} or function
        The combining function to use. Default is "fisher".
        Valid combining functions must take in p-values as their argument and be
        monotonically decreasing in each p-value.

    Returns
    -------
    array of adjusted p-values
    """
    j = len(pvalues)
    if j < 2:
        raise ValueError("One p-value: nothing to adjust!")
    if j != distr.shape[1]:
        raise ValueError("Mismatch in number of p-values and size of distr")

    # Order the p-values
    order = np.argsort(pvalues)
    pvalues_ord = pvalues[order]
    distr_ord = distr[:, order]

    # Step down tree of combined hypotheses, from global test to test of the
    # individual hypothesis with largest p-value
    pvalues_adjusted = np.zeros(j)
    pvalues_adjusted[0] = npc(pvalues_ord, distr_ord, combine=combine, plus1=plus1)
    for jj in range(1, j-1):
        next_pvalue = npc(pvalues_ord[jj:], distr_ord[:, jj:], combine=combine, plus1=plus1)
        pvalues_adjusted[jj] = np.max([next_pvalue, pvalues_adjusted[jj-1]])
    pvalues_adjusted[j-1] = np.max([pvalues_ord[j-1], pvalues_adjusted[j-2]])
    pvalues_adjusted = pvalues_adjusted[np.argsort(pvalues)]
    return pvalues_adjusted


# Randomization functions

def randomize_group(data):
    r"""
    Unstratified randomization

    Parameters
    ----------
    data : Experiment object

    Returns
    -------
    Experiment object
        Experiment object with randomized group assignments
    """
    data.group = random_sample(data.group, len(data.group), prng=data.randomizer.prng)
    return data


def randomize_in_strata(data):
    r"""
    Stratified randomization where first covariate is the stratum 

    Parameters
    ----------
    data : Experiment object

    Returns
    -------
    Experiment object
        Experiment object with randomized group assignments
    """
    # first covariate is the stratum
    strata = data.covariate[:, 0]
    unique_strata = np.unique(strata)
    for value in unique_strata:
        data.group[strata == value] = random_sample(data.group[strata == value], 
                                                    len(data.group[strata == value]), 
                                                    prng=data.randomizer.prng)
    return data


# Experiment class

class Experiment():
    r"""
    A class to represent an experiment.

    Attributes
    ----------
    group : vector
        group assignment for each observation
    response : array_like
        array of response values for each observation
    covariate : array_like
        array of covariate values for each observation
    randomizer : Randomizer object
        randomizer to use when randomizing group assignments.
        default is unstratified randomization, randomize_group
    """
    def __init__(self, group = None, response = None, covariate = None, randomizer = None):
        self.group = None if group is None else np.array(group, dtype = object)
        self.response = None if response is None else np.array(response, dtype = object) 
        self.covariate = None if covariate is None else np.array(covariate, dtype = object)
        if randomizer is None:
            self.randomizer = Experiment.Randomizer(randomize = randomize_group)
        elif isinstance(randomizer, Experiment.Randomizer):
            self.randomizer = randomizer
        else:
            raise ValueError("Not of class Randomizer")
        
        
    def __str__(self):
        return "This experiment has " + str(len(self.group)) + " subjects, " + str(len(self.response[0])) \
    + " response variables, and " \
    + (str(len(self.covariate[0])) if self.covariate is not None else str(0)) \
    + " covariates."
    
    
    def randomize(self, in_place = True, seed = None):
        # reset seed, if seed not None
        if seed is not None:
            self.randomizer.reset_seed(seed)
        if in_place:
            randomized_self = self.randomizer.randomize(self)
        else:
            # make deep copy
            randomized_self = copy.deepcopy(self)
            randomized_self.randomizer.randomize(randomized_self)
        return randomized_self

    
    @classmethod
    def make_test_array(cls, func, indices):
        def create_func(index):
            def new_func(data):
                return func(data, index)
            return new_func
        test = [create_func(index) for index in indices]
        return test
    
    
    class TestFunc:
        def mean_diff(self, index):
            # get unique groups
            groups = np.unique(self.group)
            if len(groups) != 2:
                raise ValueError("Number of groups must be two")
            # get mean for each group
            mx = np.mean(self.response[:, index][self.group == groups[0]])
            my = np.mean(self.response[:, index][self.group == groups[1]])
            return mx-my
        
        def ttest(self, index):
            # get unique groups
            groups = np.unique(self.group)
            if len(groups) != 2:
                raise ValueError("Number of groups must be two")
            t = ttest_ind(self.response[:, index][self.group == groups[0]], 
                             self.response[:, index][self.group == groups[1]], equal_var=True)[0]
            return t
        
        def one_way_anova(self, index):
            tst = 0
            overall_mean = np.mean(self.response[:, index])
            for k in np.unique(self.group):
                group_k = self.response[:, index][self.group == k]
                group_mean = np.mean(group_k)
                nk = len(group_k)
                tst += (group_mean - overall_mean)**2 * nk
            return tst
    
    
    class Randomizer():
        def __init__(self, randomize = randomize_group, seed = None):
            self.randomize = randomize
            self.prng = get_prng(seed) 
            
        # reset seed
        def reset_seed(self, seed = None):
            self.prng = get_prng(seed)
