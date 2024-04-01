import pytest

import numpy as np
from numpy.random import RandomState
from cryptorandom.cryptorandom import SHA256
import pandas as pd

from ..experiment import Experiment

def test_data():
    # create pandas dataframe
    df = pd.DataFrame({
    'res1': np.random.randn(10), ## FIX THESE VALUES SO NOT RANDOM
    'res2': np.random.randn(10),
    'cov1': np.random.randn(10),
    'cov2': np.random.randn(10),
    'strata': np.random.randint(1, 5, 10),
    'group': [2, 1, 2, 1, 1, 1, 1, 2, 1, 2]})
    # create data object from pandas dataframe
    data_instance = Experiment.Data.from_dataframe(df, 
                                                   response_col=['res1','res2'], 
                                                   covariate_col=['cov1', 'cov2'], 
                                                   strata_col='strata', 
                                                   group_col='group')
    np.testing.assert_equal(data_instance.group, [2, 1, 2, 1, 1, 1, 1, 2, 1, 2])
    ### ADD MORE CODE HERE TO CHECK THAT DATA OBJECT IS SET UP CORRECTLY
    
    
### ADD UNIT TESTS FOR ALL FUNCTIONS