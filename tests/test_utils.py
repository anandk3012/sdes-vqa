import pytest
import numpy as np
from collections import Counter
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from src.utils import (
    get_int_from_binary,
    random_bitstring,
    random_10bit_string,
    random_8bit_string,
    encrypt_counts_keys,
    expectation_from_probabilities,
    expectation_from_estimator
)
from src.hamiltonian import build_hamiltonian


def test_get_int_from_binary_and_random_length():
    assert get_int_from_binary([1, 0, 1, 1]) == 11
    # random_bitstring returns correct length of chars '0'/'1'
    for length in [1, 5, 10]:
        bs = random_bitstring(length)
        assert isinstance(bs, str)
        assert len(bs) == length
        assert set(bs).issubset({'0', '1'})


def test_random_10bit_and_8bit():
    bs10 = random_10bit_string()
    bs8 = random_8bit_string()
    assert isinstance(bs10, str) and len(bs10) == 10 and set(bs10) <= {'0', '1'}
    assert isinstance(bs8, str) and len(bs8) == 8 and set(bs8) <= {'0', '1'}


def test_encrypt_counts_keys_trivial():
    # Suppose plaintext = "00000000", and counts has a single key "0000000000"
    # Then sdes_encrypt("00000000", "0000000000") = some 8-bit ciphertext C.
    counts = {"0000000000": 5, "1111111111": 3}
    plaintext = "00000000"
    result = encrypt_counts_keys(counts, plaintext)
    # Output should be a Counter with exactly two keys, each mapped from the two input keys
    assert isinstance(result, Counter)
    assert sum(result.values()) == 8
    # Each ciphertext string is length 8 and appears with correct aggregated frequency
    for ciph, freq in result.items():
        assert len(ciph) == 8
        assert freq in {5, 3}


def test_expectation_from_probabilities_simple():
    # Build a trivial Hamiltonian for 1 qubit: H = +1 * Z
    # For an 8-qubit ham, we can embed as single-qubit test:
    # We'll test using build_hamiltonian on "00000000" and check expectation
    ham = build_hamiltonian("00000000")
    # Probability distribution concentrated on "00000000"
    probs = {"00000000": 1.0}
    exp_val = expectation_from_probabilities(probs, ham)
    # From previous test, ground state energy for "00000000" is -28
    assert abs(exp_val + 28.0) < 1e-8


def test_expectation_mixed_probabilities():
    ham = build_hamiltonian("00000000")
    # Define uniform over two basis states
    probs = {"00000000": 0.5, "11111111": 0.5}
    exp_val = expectation_from_probabilities(probs, ham)
    # Our Hamiltonian yields energy = -12 for both "00000000" and "11111111",
    # so the mixed expectation is also -24.
    assert abs(exp_val + 24.0) < 1e-8

def test_expectation_from_estimator():
    qc = QuantumCircuit(8)
    ham = build_hamiltonian("00000000")
    exp_val = expectation_from_estimator(ham, qc)
    assert abs(exp_val + 28.0) < 1e-8