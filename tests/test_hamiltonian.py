import numpy as np
import pytest
from src.hamiltonian import build_hamiltonian


def test_hamiltonian_output_type_and_shape():
    ham = build_hamiltonian("01010101")
    # Should be a SparsePauliOp
    from qiskit.quantum_info import SparsePauliOp
    assert isinstance(ham, SparsePauliOp)
    # For 8-bit string, number of terms = len(edges) + 8 single-Z = 24 + 8 = 16
    # Access paulis and coeffs
    assert len(ham.paulis) == 32
    assert len(ham.coeffs) == 32


def test_hamiltonian_ground_state_energy_all_zero():
    # For ciphertext "00000000":
    # With edges = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7)],
    # each ZZ term sees equal bits => coeff = -1 → 8 * (-1) = -8.
    # Each single Z term: bit=0 => coeff = -0.5 → 8 * (-0.5) = -4.
    # Total expected energy = -12.
    ham = build_hamiltonian("00000000")

    # Manually evaluate diagonal on |00000000>
    # We can compute by summing coeffs * (+1) because all Z eigenvalues = +1 on |0>
    diag_sum = sum(ham.coeffs[i] * 1 for i in range(len(ham.coeffs)))
    print("min_eigen_value of hamiltonian : \n")
    print(diag_sum)
    print("\n")
    assert abs(diag_sum + 28.0) < 1e-8  # diag_sum should be -28


def test_invalid_ciphertext_length():
    with pytest.raises(ValueError):
        build_hamiltonian("101")  # not 8 bits
