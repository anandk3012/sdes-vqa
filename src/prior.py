import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def gaussian_prior(n_bits: int, center_key: np.ndarray, std_dev: float) -> np.ndarray:
    """
    Construct a normalized Gaussian prior over all 2^n_bits possible keys,
    where distance is measured by Hamming distance to center_key.

    Args:
        n_bits: Number of bits in the key (e.g., 10).
        center_key: A length-n_bits numpy array of 0s and 1s indicating the center of the prior.
        std_dev: Standard deviation for the Gaussian on Hamming distance.

    Returns:
        A length-(2^n_bits) numpy array of probabilities summing to 1.
    """
    # Enumerate all 2^n_bits keys as bit arrays
    all_keys = np.array([list(map(int, format(i, f"0{n_bits}b"))) for i in range(2**n_bits)])
    # Compute Hamming distance to center_key for each key
    hamming_dist = np.sum(all_keys != center_key, axis=1)
    # Evaluate Gaussian PDF at those distances (mean=0)
    probs = norm.pdf(hamming_dist, 0, std_dev)
    # Normalize into a probability distribution
    return probs / np.sum(probs)


def hamming_distance(str1: str, str2: str) -> int:
    """
    Compute the Hamming distance between two equal-length bitstrings.

    Args:
        str1: First bitstring (e.g., "1010010101").
        str2: Second bitstring of the same length.

    Returns:
        The number of positions at which the corresponding bits differ.
    """
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length.")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def cosine_distance(x, y):
    """Cosine distance = 1 - cosine similarity"""
    dot = np.dot(x, y)
    norm_product = np.linalg.norm(x) * np.linalg.norm(y)
    if norm_product == 0:
        return 1.0  # define max distance if either is zero vector
    return 1 - (dot / norm_product)

def cosine_gaussian_prior(n_bits: int, center_key: np.ndarray, std_dev: float) -> np.ndarray:
    """
    Construct a Gaussian prior over 2^n_bits binary keys based on cosine distance.
    
    Args:
        n_bits: Number of bits per key.
        center_key: Binary array of length n_bits representing the center.
        std_dev: Standard deviation for the Gaussian kernel over cosine distance.
    
    Returns:
        Normalized prior (length 2^n_bits) as numpy array.
    """
    num_keys = 2**n_bits
    all_keys = np.array([list(map(int, format(i, f"0{n_bits}b"))) for i in range(num_keys)])
    
    # Compute cosine distances
    distances = np.array([cosine_distance(key, center_key) for key in all_keys])
    
    # Apply Gaussian kernel: exp(-0.5 * (distance / std_dev)^2)
    unnormalized = np.exp(-0.5 * (distances / std_dev)**2)
    prior = unnormalized / np.sum(unnormalized)
    
    return prior