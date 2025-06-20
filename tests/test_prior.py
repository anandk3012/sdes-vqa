import numpy as np
import pytest
from src.prior import gaussian_prior, hamming_distance


def test_prior_sums_to_one():
    # Center at all-zero key
    center = np.zeros(10, dtype=int)
    prior = gaussian_prior(n_bits=10, center_key=center, std_dev=2.0)
    assert prior.shape == (2**10,)
    assert abs(np.sum(prior) - 1.0) < 1e-8


def test_hamming_distance_basic():
    assert hamming_distance("0000", "0000") == 0
    assert hamming_distance("1010", "0101") == 4
    assert hamming_distance("1100", "1001") == 2

    with pytest.raises(ValueError):
        hamming_distance("101", "10")  # unequal lengths
