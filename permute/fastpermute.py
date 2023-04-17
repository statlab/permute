"""
This function implements Aronow et al "Fast computation of exact confidence intervals for randomized
experiments with binary outcomes" algortihm 4.3
"""
import numpy as np

def Random_Sample(n: int, k: int, gen: callable=np.random) -> set: 
    # from Cormen et al.
    if k==0:
        return set()
    else:
        S = Random_Sample(n-1, k-1)
        i = gen.randint(1,n+1) 
        if i in S:
            S = S.union([n])
        else:
            S = S.union([i])
    return S

def simulate_obs_n(w: list, m: int, l: int = 1e4, gen: callable=np.random):
    ''' 
    Generate l tables given w, m, and prng
    
    Parameters
    ----------
    w  : list of 4 ints
        the table of counts of subjects with each combination of potential outcomes, in the order
        [w11, w10, w01, w00]
        
    m : int
        number of subjects to be assigned to the active treatment
    
    l  : int
        number of replication
    
    gen: RandomState
        use to generare pseudo-random samples
    
    Returns
    -------
    tables : l lenghth list, where each list contains 4 ints of the simulated tables
        [[n11, n10, n01, n00], ...]
    '''
    l = int(l)
    units = np.sum(w)
    tables= []
    # Note this is the cumulative sum, different from how we usually define N's
    w = w[::-1]
    N00, N01, N10, N00 = np.cumsum(w)
    
    # repeat l times
    for i in range(l):
        # Generate random number
        treat = np.array(list(Random_Sample(units, m, gen)))
        # Count how many in each group is assignment to treatment
        count = [np.sum(treat<=N00), np.sum((treat<=N01)&(treat>N00)), 
                 np.sum((treat<=N10)&(treat>N01)), np.sum(treat>N10)]
        # Find the simulated tables
        tables += [count[::-1]]
    return tables

def permtest(v: list, n: list, l: int = 1e6, gen: callable=np.random) -> float:
    '''
    This function implements aronow et al "Fast computation of exact confidence intervals for randomized
experiments with binary outcomes" algortihm 2.2.
    
    return a p value under permutation test
    
    Parameters:
    ----------
    v : list of four ints
        the table of counts of subjects with each combination of potential outcomes, in the order
        [N00, N01, N10, N11]
       
    n  : list of four ints
        the table of observed outcome
        [n00, n01, n10, n11]
            
    Returns:
    --------
    p-value : bool
        p_value of the permuation test
    '''
    n11, n10, n01, n00 = n
    v11, v10, v01, v00 = v
    n = np.sum(v)
    l = int(l)
    
    # Find test statistic of v, n
    def t(w):
        m = w[1] + w[0]
        n = np.sum(w)
        if m == 0:
            return - w[2]/(w[3] + w[2])
        elif m==n:
            return w[0]/(w[1] + w[0]) 
        else:
            return w[0]/(w[1] + w[0]) - w[2]/(w[3] + w[2])
            
    tv = t(v)
    tn = t([n11, n10, n01, n00])
    # simulate l times
    tables = simulate_obs_n([n11, n10, n01, n00], n10 + n11)
    simulated_t = np.apply_along_axis(t, 1, tables)
    
    # Calculate p-value  
    p = np.mean(np.abs(simulated_t - tv) >= np.abs(tn - tv))
    return p
    

def fastcompuation(n: list, alpha = 0.05):
    ''' 
    This function implements aronow et al "Fast computation of exact confidence intervals for randomized
experiments with binary outcomes" algortihm 4.3

    
    Parameters
    ----------
        
    n : list of 4 ints
        the observed outcome
        [n11, n10, n01, n00]
        
    alpha: float
        the significance level
    
    Returns
    -------
    (lower, upper): the lower and upper bound of the confidence set
    '''
    n11, n10, n01, n00 = n
    n = sum(n)
    tn = n11/(n10 + n11) - n01/(n00 + n01)

    def f(t0):
        '''
        Return if t0 is compatible with the observed outcome n 
        '''
        count = 0
        j = max(n*t0+n01, n11)
        while j <= min(n-n10, n11+n*t0+n10+n01):
            maxv10 = min([j, n11+n00, n10+n01+n*t0, n+n*t0-j])
            minv10 = max([0, n*t0, j-n11-n01, n11+n01+n*t0-j])
            if (minv10 > maxv10):
                j+=1
                continue
                
            v10 = minv10
            v00 = n - j - v10 + n*t0
            v11 = j - v10
            v01 = v10 - n*t0
            
            if (v11 == 0 and v10 == 0):
                v11 = 1
            v = [v11, v10, v01, v00]
            if (any(np.array(v) < 0)):
                j+=1
                continue
            p = permtest(v, [n11, n10, n01, n00])
            count += 1
            if p >= alpha:
                return 0
            j+=1
        return 1
                      

    k1 = round(n*tn)+1
    k2 = n

    
    upper = -1
    lower = 1
    # binary search for upper bound
    a = k1
    b = k2
    c = (a+b)//2
    out = f(c)
    while b > a+1:
        out = f(c/n)
        if out == 0:
            a = c
        else:
            b = c
        c = (a+b)//2
    
    if a > k1 and b < k2:
        upper =  a
    elif a == k1:
        if f(k1/n) == 1:
            upper =  k1-1
        else:
            upper =  k1
    elif b == k2:
        if f(k2/n) == 1:
            upper =  k2-1
        else:
            upper =  k2
    
    # binary search for lower bound
    a = -k2
    b = k1
    c = (a+b)//2
    k1 = -k2
    k2 = b
    
    while b > a+1:
        out = f(c/n)
        if out == 0:
            a = c
        else:
            b = c
        c = (a+b)//2
    if a > k1 and b < k2:
        lower =  a
    elif a == k1:
        if f(k1/n) == 1:
            lower =  k1+1
        else:
            lower =  k1
    elif b == k2:
        if f(k2/n) == 1:
            lower =  k2+1
        else:
            lower =  k2
            
    return (lower, upper)

