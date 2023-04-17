from ..gaffke import gaffke_bound
import pytest
import numpy as np



def test_lower_bound_property():
    """
    Test the conservative property of the Gaffke bound by checking
    if the lower bound is less than or equal to the true mean for
    different sample sizes and distributions.
    """
    np.random.seed(42)  # Set the random seed for reproducibility

    # Test parameters
    sample_sizes = [10, 50, 100]
    distributions = [
        {"type": "uniform", "params": (0, 1)},
        {"type": "exponential", "params": (0.5,)},
        {"type": "normal_positive", "params": (1, 0.5)},
    ]

    for dist in distributions:
        for size in sample_sizes:
            if dist["type"] == "uniform":
                samples = np.random.uniform(dist["params"][0], dist["params"][1], size)
                true_mean = np.mean(samples)
            elif dist["type"] == "exponential":
                samples = np.random.exponential(dist["params"][0], size)
                true_mean = np.mean(samples)
            elif dist["type"] == "normal_positive":
                samples = np.abs(np.random.normal(dist["params"][0], dist["params"][1], size))
                true_mean = np.mean(samples)

            lower_bound = gaffke_bound(samples)

            assert(lower_bound <= true_mean)

