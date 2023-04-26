"""
This function implements Aronow et al "Fast computation of exact confidence intervals for randomized
experiments with binary outcomes" algortihm 4.3
"""
def t(w: list):
    ''' 
    Find the average treatement effect
    
    Parameters
    ----------
    w : list of 4 ints
        the observed outcome
        [w11, w10, w01, w00]
        
    Returns
    -------
    average treatement effect
    '''

    m = w[0] + w[1]
    n = np.sum(w)
    if m == 0:
        return - w[2]/(w[2] + w[3])
    elif m==n:
        return w[0]/(w[0] + w[1]) 
    else:
        return (w[0]/(w[0] + w[1])) - (w[2]/(w[2] + w[3]))

def findp(N, n, l=int(1e4), seed = 42):#, debug=False):
    ''' 
    Find the p value of whether potential outcome table N is consistent with obersved table n
    
    Parameters
    ----------
    n : list of 4 ints
        the observed outcome
        [n11, n10, n01, n00]
        
    N : list of four ints
        the table of counts of subjects with each combination of potential outcomes, in the order
        [N11, N10, N01, N00]
        as defined in Aronow et. al.
        
    Returns
    -------
    p value of whether potential outcome table N is consistent with obersved table n
    '''
    
    np.random.seed(seed)
    obs = t(n)
    tN = (N[1] - N[2])/sum(N)
    N11, N10, N01, N00 = np.cumsum(N)
    k = n[0]+n[1]
    cnt = 0
    for i in range(l):
        treat = np.random.choice(N00, k, replace=False)
        count = [np.sum(treat<N11), np.sum((treat<N10)&(treat>=N11)), 
                 np.sum((treat<N01)&(treat>=N10)), np.sum(treat>=N01)]
        n11 = count[0]*1 + count[1]*1 + count[2]*0 + count[3]*0
        n10 = count[0]*0 + count[1]*0 + count[2]*1 + count[3]*1
        n01 = (N[0] - count[0])*1 + (N[1]-count[1])*0 + (N[2]-count[2])*1 + (N[3]-count[3])*0
        n00 = (N[0] - count[0])*0 + (N[1]-count[1])*1 + (N[2]-count[2])*0 + (N[3]-count[3])*1
        tn = t([n11, n10, n01, n00])
        if (abs(tn-tN) >= abs(obs - tN)):
            cnt += 1
    return (cnt+1)/(l+1) 

def permutation_test(n, alpha, Tn):
    ''' 
    Conduct permutation test. 
    
    Parameters
    ----------
    n : list of 4 ints
        the observed outcome
        [n11, n10, n01, n00]
        
    alpha: significant level
        
    Returns
    -------
    0 if accept, 1 if reject
    '''
    N = sum(n)
    n11, n10, n01, n00 = n
    cnt = 0
    for j in range(N+1):
        if j >= int(N*Tn)+n01 and j >= n11 and N >= j+n10 and n11+int(N*Tn)+n10+n01 >= j:
            lower_bound = max(0, int(N*Tn), j-n11-n01, n11+n01+int(N*Tn)-j)
            upper_bound = min(j, n11+n00, n10+n01+int(N*Tn), N+int(N*Tn)-j)
            for v10 in range(lower_bound, upper_bound+1):
                v = (j-v10, v10, v10-int(N*Tn), N-j-v10+int(N*Tn))
                p_value = findp(v, n)
                cnt += 1
                if p_value >= alpha:
                    return 0
                elif v[1] == 0 and v[2] == 0:
                    p_value = findp([v[0]-1, v[1]+1, v[2], v[3]], n)

                    cnt += 1
                    if p_value >= alpha:
                        return 0
    return 1

def binary_search(k1, k2, f):
    ''' 
    Binary search for the upper bound of confidence interval
    
    Parameters
    ----------
    k1: left end point
    
    k2: right end point
    
    f: function
        
    Returns
    -------
    int, the upper bound
    '''
    a = k1
    b = k2
    while b > a + 1:
        c = math.floor((a + b) / 2)
        #print("binary bound:", a, b, c)
        if f(c) == 0:
            a = c
        else:
            b = c
    #print("binary bound:", a, b, c)
    if a == k1:
        if f(k1) == 0:
            return k1
        else:
            return k1 - 1
    elif b == k2:
        if f(k2) == 0:
            return k2
        else:
            return k2 - 1
    else:
        return a
    
def binary_search_opp(k1, k2, f):
    ''' 
    Binary search for the lower bound of confidence interval
    
    Parameters
    ----------
    k1: left end point
    
    k2: right end point
    
    f: function
        
    Returns
    -------
    int, the lower bound
    '''
    a = k1
    b = k2
    while b > a + 1:
        c = math.ceil((a + b) / 2)
        #print("binary bound:", a, b, c)
        if f(c) == 0:
            b = c
        else:
            a = c
    #print("binary bound:", a, b, c)
    if b == k2:
        if f(k2) == 0:
            return k2
        else:
            return k2 + 1
    elif a == k1:
        if f(k1) == 0:
            return k1 
        else:
            return k1 + 1
    else:
        return b

def find_interval(alpha, n):
    N = sum(n)
    n11, n10, n01, n00 = n
    Tn = t(n)
    def f(x):
        return permutation_test(n, alpha, x/N)
    
    k2 = round(N*Tn)
    k1 = n11+n00-N 
    #print("k1", k1, "k2", k2)
    L = binary_search_opp(k1, k2, lambda x: f(x))
    
    k1 = round(N*Tn)
    k2 = n01 + n10#n00 + n11 #the original stuff is n01 + n10
    #print("k1", k1, "k2", k2)
    U = binary_search(k1, k2, lambda x: f(x))
    return N*Tn, np.array([L, U])
