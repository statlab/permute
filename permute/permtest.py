import .experiment

# Run permutation test
def permutation_test(exper, reps=int(10**4), keep_dist = False, keep_randomization = False, 
                     lockstep = False):
    r"""
    A function to run a permutation test
    
    Attributes
    ----------
    exper : Experiment object
    reps : int
        number of permutations that the permutation test should use
    keep_dist : bool
        if true, return the test statistic distribution, default False
    keep_randomization : bool
        if true, return the allocation of units under each randomization, default False
    lockstep : bool
        if true, use the same set of permutations for every response variable, default False
    """  
    # check data is of type Experiment
    if not isinstance(exper, Experiment):
        raise ValueError("data not of class Experiment")
        
    data = exper.data
    test = exper.testfunc
    randomizer = exper.randomizer
    
    # if keep randomization, make sure return allocation set to true in randomizer
    if keep_randomization:
        randomizer.return_allocation = True
        allocations = {c: [] for c in range(len(test))} if not lockstep else {'lockstep': []}
    
    ts = {}
    tv = {}
    ps = {}

    # get the test statistic for each column on the original data
    for c in range(len(test)):
        # apply test statistic function to column
        ts[c] = test[c](data)
        tv[c] = []
        
    # check if randomization in place
    if randomizer.in_place:
        data_copy = data
    else:
        data_copy = copy.deepcopy(data)
        
    # get test statistics for random samples
    for i in range(reps):
        if lockstep:
            # if lockstep, randomly permute group
            rand_output = randomizer.randomize(randomizer, data = data_copy)
            if keep_randomization:
                allocations['lockstep'].append(rand_output)  
        # calculate test statistics on permuted data
        for c in range(len(test)):
            # if independent randomizations, randomize for each test
            if not lockstep:
                rand_output = randomizer.randomize(randomizer, data = data_copy)
                if keep_randomization:
                    allocations[c].append(rand_output) 
            # get test statistic for this permutation
            tv[c].append(test[c](data_copy))
    # get p-values for original data
    for c in range(len(test)):
        ps[c] = (np.sum(np.array(tv[c]) >= ts[c]) + 1)/(reps + 1)  
    # save dist if keep_dist, otherwise dist = None    
    if keep_dist:
        # change format of dist to array
        dist = np.array([tv[c] for c in range(len(test))]).T
        # append test statistic from orignal data to dist
        dist = np.append(dist, np.array([ts[c] for c in range(len(test))], ndmin=2), axis=0)
    else:
        dist = None
        
    return (ts, ps, dist, allocations) if keep_randomization else (ts, ps, dist)