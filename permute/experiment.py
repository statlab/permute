import numpy as np
from numpy.random import RandomState
import copy
from scipy.stats import norm, rankdata, ttest_ind, ttest_1samp
import cryptorandom as cr
from cryptorandom.sample import random_allocation, random_sample
from .utils import get_prng, permute
import pandas as pd

# Experiment class
class Experiment():
    r"""
    A class to represent an experiment, including the data, experimental design (randomizer), and test statistic

    Attributes
    ----------
    data : Data object
    randomizer : Randomizer object
        randomizer to use when randomizing group assignments.
    testfunc : Testfunc object
    """    
    def __init__(self, data=None, randomizer=None, testfunc=None):
        self.data = data
        self.randomizer = randomizer
        self.testfunc = testfunc    
        
    def __str__(self):
        # compute number of covariates
        if self.data.covariate is None:
            covariate_num = 'no'
        elif isinstance(self.data.covariate[0], np.ndarray):
            covariate_num = str(len(self.data.covariate[0]))
        else:
            covariate_num = '1'
        # experiment description
        description = "This experiment has " + str(len(self.data.group)) + " subjects, " + \
        str(len(self.data.response[0]) if isinstance(self.data.response[0], np.ndarray) else '1') + \
        " response variable(s), " + covariate_num + " covariate(s), and "  + \
        str('no' if self.data.strata is None else len(np.unique(self.data.strata))) + " strata."

        return description

    class Data:
        r"""
        A class to represent data

        Attributes
        ----------
        group : array_like
            group assignment for each observation
        response : array_like
            array of response values for each observation
        strata : array_like
            strata assignment for each observation
        covariate : array_like
            array of covariate values for each observation
        """  
        def __init__(self, group = None, response = None, strata = None, covariate = None):
            self.group = None if group is None else np.array(group, dtype = object)
            self.response = None if response is None else np.array(response, dtype = object) 
            self.strata = None if strata is None else np.array(strata, dtype = object) 
            self.covariate = None if covariate is None else np.array(covariate, dtype = object)

        
        @classmethod
        def from_dataframe(cls, df, response_col=None, covariate_col=None, strata_col=None, group_col=None):
            response = df[response_col].to_numpy() if response_col else None
            covariate = df[covariate_col].to_numpy() if covariate_col else None
            strata = df[strata_col].values if strata_col else None
            group = df[group_col].values if group_col else None
            
            return cls(response=response, covariate=covariate, strata=strata, group=group)
        
        
    
    @classmethod
    def make_test_array(cls, func, indices):
        def create_func(index):
            def new_func(data):
                return func(data, index)
            return new_func
        test = [create_func(index) for index in indices]
        return test
    
    
    class TestFunc:

        @classmethod
        def mean_diff(cls, data, index):
            groups = np.unique(data.group)
            if len(groups) != 2:
                raise ValueError("Number of groups must be two")
            mx = np.mean(data.response[:, index][data.group == groups[0]])
            my = np.mean(data.response[:, index][data.group == groups[1]])
            return mx - my
        
        @classmethod
        def ttest(cls, data, index):
            # get unique groups
            groups = np.unique(data.group)
            if len(groups) != 2:
                raise ValueError("Number of groups must be two")
            t = ttest_ind(data.response[:, index][data.group == groups[0]],
                      data.response[:, index][data.group == groups[1]],
                      equal_var=True)[0]
            return t
        
        @classmethod
        def one_way_anova(cls, data, index):
             tst = 0
             overall_mean = np.mean(data.response[:, index])
             for k in np.unique(data.group):
                  group_k = data.response[:, index][data.group == k]
                  group_mean = np.mean(group_k)
                  nk = len(group_k)
                  tst += (group_mean - overall_mean) ** 2 * nk
             return tst
    
    ## To do: add test statistic for one_sample
    
    class Randomizer():
        def __init__(self, randomize: callable = None, seed : object = None, return_allocation : bool = False,
                     in_place : bool = True, **kwargs):
            self.randomize = randomize
            self.prng = get_prng(seed) 
            self.return_allocation = return_allocation
            self.in_place = in_place
            self.kwargs = kwargs 
            self.__dict__.update(kwargs)
            
        # reset seed
        def reset_seed(self, seed : object = None):
            self.prng = get_prng(seed)

        def randomize_group(self, data):
            r"""
            Stratified randomization of group assignments 

            Parameters
            ----------
            exper: Experiment
                instance of the Experiment class to randomize
            return_allocation : boolean
                return a dict of items assigned to each group?    
            in_place : boolean
                randomize the assignment within the Experiment object, or return a new Experiment object?

            Returns
            -------
            Experiment object with randomized group assignments if `not in_place`
            Allocation if `return_allocation`

            Side Effects
            ------------
            Randomizes the group assignments of the Experiment object if `in_place`

            """
            if self.in_place:
                if data.strata is None:
                    data.group = random_sample(data.group, len(data.group), prng=self.prng)
                else:
                    unique_strata = np.unique(data.strata)
                    for value in unique_strata:
                        data.group[data.strata == value] = random_sample(data.group[data.strata == value], 
                                                                    len(data.group[data.strata == value]), 
                                                                    prng=None)
            else:
                # if not in place make copy of data
                new_data = copy.deepcopy(data)
                if new_data.strata is None:
                    new_data.group = random_sample(new_data.group, len(new_data.group), prng=self.prng)
                else:
                    unique_strata = np.unique(new_data.strata)
                    for value in unique_strata:
                        new_data.group[new_data.strata == value] = random_sample(new_data.group[new_data.strata == value], 
                                                                    len(new_data.group[new_data.strata == value]), 
                                                                    prng=self.prng)
            # if return_allocation, generate allocation dict
            if self.return_allocation:
                group_counts = np.unique(data.group, return_counts = True)
                allocation = {}
                for g in group_counts[0]:
                    allocation[g] = np.where(data.group == g)[0]
                returned = allocation if self.in_place else (new_data, allocation)
            else:
                returned = None if self.in_place else new_data

            return returned

        
        def randomize_multinomial(self, data, **kwargs):
            r"""
            Multinomial randomization of group assignments.

            Parameters
            ----------
            self : Experiment object
                Experiment object with group assignments.
            p : dict
                keys are group labels; values are floats, the probability of assigning the corresponding label
                The sum of the probabilities must be 1.

            Returns
            -------
            Experiment object
                Experiment object with multinomial randomized group assignments if `not in_place`

            Side Effects
            ------------
            Randomizes the group assignments of the Experiment object if `in_place`

            """
            p = self.kwargs.get('p')

            # Ensure the sum of the probability dict is 1
            if not np.isclose(np.sum(list(p.values())), 1): 
                  raise ValueError(f'The probabilities sum to {np.sum(p.values())}')

            group_labels = list(p.keys())
            probs = list(p.values())

            # multinomial randomization 
            group_assignments = self.prng.choices(group_labels, k=len(data.group), weights = probs)

            # allocation dict: keys are group labels; values are indices of items assigned to that group
            a = {g : np.where(group_assignments == g)[0] for g in group_labels}

            # Update the experiment object's group assignments
            if self.in_place:
                data.group = group_assignments
                returned = a if self.return_allocation else None
            else:
                new_data = copy.deepcopy(data)
                new_data.group = group_assignments
                returned = (new_data, a) if self.return_allocation else new_data
            return returned