#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions used to compute the CI for ATE using Li and Ding approach

@author: Qiyu Wang
"""

# import necessary package
import numpy as np

######### helper functions #########
def filter_table(Nt: list, n00: int, n01: int, n10: int, n11: int) -> bool:
    '''
    check whether summary table Nt of binary outcomes is consistent with observed counts
    
    implements the test in Theorem 1 of Li and Ding (2016)
    
    Parameters:
    ----------
    Nt : list of four ints
        the table of counts of subjects with each combination of potential outcomes, in the order
        N_00, N_01, N_10, N_11
        
    n00: int
        number of subjects assigned to control whose observed response was 0
       
    n01 : int
        number of subjects assigned to control whose observed response was 1
        
    n10: int
        number of subjects assigned to treatment whose observed response was 0

    n11 : int
        number of subjects assigned to treatment whose observed response was 1
        
        
    Returns:
    --------
    ok : bool
        True if table is consistent with the data
    '''
    N = np.sum(Nt)   # total subjects
    return max(0,n11-Nt[1], Nt[3]-n01, Nt[2]+Nt[3]-n10-n01) <= min(Nt[3], n11, Nt[2]+Nt[3]-n01, N-Nt[1]-n01-n10)


def N_generator(N: int, n00: int, n01: int, n10: int, n11: int) -> tuple:
    ''' 
    generate tables algebraically consistent with data from an experiment with binary outcomes
    
    Parameters
    ----------
    N : int
        number of subjects
    n00 : int
        number of subjects assigned to treatment 0 who had outcome 0
    n01 : int
        number of subjects assigned to treatment 0 who had outcome 0
    n10 : int
        number of subjects assigned to treatment 1 who had outcome 0
    n11 : int
        number of subjects assigned to treatment 1 who had outcome 1
    
    Returns
    -------
    Nt : list of 4 ints 
        N00, subjects with potential outcome 0 under treatments 0 and 1
        N01, subjects with potential outcome 0 under treatment 0 and 1 under treatment 1
        N10, subjects with potential outcome 1 under treatment 0 and 0 under treatment 1
        N11, subjects with potential outcome 1 under treatments 0 and 1
    '''
    for i in range(min(N-n00, N-n10)+1):               # allocate space for the observed 0 outcomes, n00 and n10
        N11 = i                                           
        for j in range(max(0, n01-N11), N-n00-N11):    # N11+N10 >= n01; N11+N10+n00 <= N
            N10 = j                                        
            for k in range(max(0, n11-N11), min(N-n10-N11, N-N11-N10)): 
                                                       # N11+N01 >= n11; N11+N01+n10 <= N; no more than N subjects
                N01 = k                                  
                N00 = N-N11-N10-N01                  
                if filter_table([N00, N01, N10, N11], n00, n01, n10, n11):
                    yield [N00, N01, N10, N11]
                else:
                    pass
   
                
def sim_obs(Nt:list,n:int,rep:int,rng):
    '''
    simulate observed response data from hypothesized outcome table
    
    Parameters:
    ----------------------------------------
    Nt: a list of four numbers with each combination of potential outcomes
    N_00, N_01, N_10, N_11
    
    n: an int indicating the number of subjects receiving treatment
    
    rep: the number of replications
    
    rng: an object of class RandomState
    
    Returns:
    ----------------------------------------
    a list of four int
    n_00, n_01, n_10, n_11
    '''
    N = sum(Nt) # the number of subjects we have
    for r in np.arange(rep):
        ind_tr = rng.choice(N,n,replace=False) # randomly generate the index of subjects receiving treatment
        ind_cont = np.array([i for i in np.arange(N) if i not in ind_tr]) # the remaining N-n subjects are assigned control
        n_00 = 0
        n_01 = 0
        n_10 = 0
        n_11 = 0
    
        for i in ind_tr:
            if i < Nt[0] or (i >= Nt[0]+Nt[1] and i < Nt[0]+Nt[1]+Nt[2]):  # if the subject's respond is 0 to treatment
                n_10+=1
            else: # if the subject's respond is 1 to treatment
                n_11+=1
    
        for i in ind_cont:
            if i < Nt[0]+Nt[1]:
                n_00+=1
            else:
                n_01+=1
        
        yield [n_00,n_01,n_10,n_11]
   
        
def T(Nt,nt): 
    '''
    the sample test statistics from Li and Ding
    the absolute difference between true ATE and the one from the unbiased estimator
    
    Parameters:
    ------------------------------------
    Nt: a list of four numbers with each combination of potential outcomes
    N_00, N_01, N_10, N_11
    
    nt: a list of four numbers with each combination of observed responses
    n_00, n_01, n_10, n_11
    '''
    return abs(nt[3]/(nt[2]+nt[3])-nt[1]/(nt[0]+nt[1])-(Nt[1]-Nt[2])/(Nt[0]+Nt[1]+Nt[2]+Nt[3]))


def stat_consist(Nt:list,nt:list,T=T,rep=10000,rng=np.random.RandomState()):
    '''
    test whether a hypothesized outcome table is statistically consistent with an observed response
    
    Parameters:
    -----------------------------------------
    Nt: a list of four numbers with each combination of potential outcomes
    N_00, N_01, N_10, N_11
    
    nt: a list of four numbers with each combination of observed responses
    n_00, n_01, n_10, n_11
    
    T: a test statistics taking Nt and nt as arguments
    
    rep: the number of iterations we used to simulate the distribution of test stats, with default value 1000
    
    rng: an object of class RandomState
    
    
    
    returns:
    -----------------------------------------
    right-tailed p value
    
    '''
    
    t = T(Nt,nt)
    n = nt[2]+nt[3]  # the number of subjects receiving treatment in the observed data
    sim_nt = sim_obs(Nt,n,rep,rng)  # simulate the distribution
    t_dist = [T(Nt,snt) for snt in sim_nt]
    t_dist.append(t)
    return len([st for st in t_dist if st >= t])/(rep+1) # return p-value, the probability we get test stats greater than we observed

############# main function #############
def ate_ci(nt:list,T=T,rep=10000,rng=np.random.RandomState(),sterne=False,level=0.95):
    '''
    find the lower and upper bound of ATE given the observed responses
    
    Parameters:
    ----------------------------------------
    nt: the observed responses
    n_00, n_01, n_10, n_11
    
    T: a test statistics takinf Nt and nt as arguments. A default one is given.
    
    rep: the number of iterations used to simulate the distribution. The default is 10^4
    
    rng: a RandomState object. A default one without seed is given
    
    sterne: indicating whether to use Sterne's method (default to be False)
    
    level: confidence level of test (default to be 0.95)
    
    returns:
    ----------------------------------------
    the lower and upper bound of CI for ATE
    '''
    ate = [] # a list of ate in the confidence set
    N = nt[0]+nt[1]+nt[2]+nt[3] # number of subjects
    Nt_list = [Nt for Nt in N_generator(N,nt[0],nt[1],nt[2],nt[3])] # generate candidate Nt
    if not sterne:
        for Nt in Nt_list: # iterate through all the potential outcome tables algebraically consistent with observations
            a = (Nt[1]-Nt[2])/N
            if len(ate)>0:
                if a > max(ate) or a < min(ate): # only test consistency if the ate is smaller than minimum or greater than maximum in the confidence set
                    if stat_consist(Nt,nt,T,rep,rng)>=1-level:
                        ate.append(a)
            else:
                if stat_consist(Nt,nt,T,rep,rng)>=1-level:
                        ate.append(a)
    else:
        p_dic_left = {} # dictionary recoridng Nt and its p-value
        p_dic_right = {}
        cur_left=0
        cur_right=0
        i = 0
        for Nt in Nt_list:
            p = stat_consist(Nt,nt,T,rep,rng)
            if p>0.5:
                p_dic_left[i] = 1-p
            else:
                p_dic_right[i] = p
            i = i+1
        while len(p_dic_left)!=0 or len(p_dic_right)!=0:
            if len(p_dic_left)==0:
                right = min(p_dic_right,key=p_dic_right.get)
                cur_right = p_dic_right[right]
                if cur_left+cur_right>1-level:
                    break
                else:
                    del p_dic_right[right]
                    continue
            if len(p_dic_right)==0:
                left = min(p_dic_left,key=p_dic_left.get)
                cur_left = p_dic_left[left]
                if cur_left+cur_right>1-level:
                    break
                else:
                    del p_dic_left[left]
                    continue
            left = min(p_dic_left,key=p_dic_left.get)
            right = min(p_dic_right,key=p_dic_right.get)
            if p_dic_left[left]<p_dic_right[right]:
                cur_left = p_dic_left[left]
                if cur_left+cur_right>1-level:
                    break
                else:
                    del p_dic_left[left]
            else:
                cur_right = p_dic_right[right]
                if cur_left+cur_right>1-level:
                    break
                else:
                    del p_dic_right[right]
        
        indices = list(p_dic_left.keys())+list(p_dic_right.keys())
        Nt_newlist = [Nt_list[x] for x in indices]
        for Nt in Nt_newlist:
            ate.append((Nt[1]-Nt[2])/N)
    if len(ate)==0:
        return np.nan,np.nan
    return min(ate),max(ate)

