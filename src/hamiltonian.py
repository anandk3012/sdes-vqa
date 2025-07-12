import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp


def build_hamiltonian(ciphertext_str: str) -> SparsePauliOp:
    
    if(len(ciphertext_str) == 16):
        return build_saes_hamiltonian(ciphertext_str)
    """
    Given an 8-bit ciphertext string, construct a Hamiltonian (SparsePauliOp)
    whose ground state is that exact bitstring.  
    We use:
      - Pairwise Z⊗Z terms over a predefined edge list; coefficient = +1 if bits differ, -1 if equal.
      - Single-qubit Z terms: +0.5 if bit='1', -0.5 if bit='0'.

    Args:
        ciphertext_str: an 8-character string of '0'/'1'.

    Returns:
        SparsePauliOp representing the Hamiltonian on 8 qubits.
    """
    if len(ciphertext_str) != 8:
        raise ValueError("ciphertext must be exactly 8 bits.")

    n = 8
    ciphertext = list(ciphertext_str)
    edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5), (1, 0),
    (2, 6), (2, 7), (2, 0),
    (3, 5), (3, 6), (3, 0),
    (4, 5), (4, 1), (4, 7),
    (5, 3), (5, 4), (5, 1),
    (6, 2), (6, 3), (6, 7),
    (7, 2), (7, 4), (7, 6)
    ]   

    # Hamiltonian
    hamiltonian_terms = []

    # pairwise z gates from graph
    n = len(ciphertext)
    for i, j in edges:
        zstring = ""
        for k in np.arange(7,-1, -1):
            if(k == i or k == j):
                zstring += 'Z'
            else:
                zstring += 'I'

        if ciphertext[n - i - 1] != ciphertext[n - j - 1]:
            hamiltonian_terms.append((1, Pauli(zstring))) # w_ij = +1
        else:
            hamiltonian_terms.append((-1, Pauli(zstring))) # w_ij = -1

    # single-qubit gates
    for i in range(n):
        zstring = ""
        for k in np.arange(7, -1, -1):
            if(k == i):
                zstring += 'Z'
            else:
                zstring += 'I'

        if ciphertext[n - i - 1] == '1':
            hamiltonian_terms.append((0.5, Pauli(zstring))) # t_i = +0.5
        else:
            hamiltonian_terms.append((-0.5, Pauli(zstring))) # t_i = -0.5

        coeffs = []
        pauliList = []

        for coeff, pauli in hamiltonian_terms:
            coeffs.append(coeff)
            pauliList.append(pauli)

    coeffs = []
    pauliList = []

    for coeff, pauli in hamiltonian_terms:
        coeffs.append(coeff)
        pauliList.append(pauli)

    # hamiltonian_ops
    hamiltonian_operator = SparsePauliOp(pauliList, coeffs) # Hermitian Hamiltonian operator

    return hamiltonian_operator


def build_saes_hamiltonian(ciphertext_str: str) -> SparsePauliOp:
    """
    Given a 16-bit ciphertext string, construct a Hamiltonian (SparsePauliOp)
    whose ground state is that exact bitstring.

    Args:
        ciphertext_str: a 16-character string of '0'/'1'.

    Returns:
        SparsePauliOp representing the Hamiltonian on 16 qubits.
    """
    if len(ciphertext_str) != 16:
        raise ValueError("ciphertext must be exactly 16 bits.")

    n = 16
    ciphertext = list(ciphertext_str)

    # Define a simple 2D grid or local connectivity (or fully connected subset)
    # For now, just use a basic connectivity: ring + short-range neighbors
    edges = []
    for i in range(n):
        if i + 1 < n:
            edges.append((i, i + 1))
        if i + 2 < n:
            edges.append((i, i + 2))

    # Hamiltonian
    hamiltonian_terms = []

    # Pairwise ZZ terms
    for i, j in edges:
        zstring = ""
        for k in reversed(range(n)):
            if k == i or k == j:
                zstring += 'Z'
            else:
                zstring += 'I'

        # Access ciphertext[i] and ciphertext[j], remembering bits are reversed
        if ciphertext[n - i - 1] != ciphertext[n - j - 1]:
            hamiltonian_terms.append((1.0, Pauli(zstring)))
        else:
            hamiltonian_terms.append((-1.0, Pauli(zstring)))

    # Single-qubit Z terms
    for i in range(n):
        zstring = ""
        for k in reversed(range(n)):
            zstring += 'Z' if k == i else 'I'

        if ciphertext[n - i - 1] == '1':
            hamiltonian_terms.append((0.5, Pauli(zstring)))
        else:
            hamiltonian_terms.append((-0.5, Pauli(zstring)))

    # Extract coefficients and Paulis
    coeffs = [c for c, _ in hamiltonian_terms]
    pauliList = [p for _, p in hamiltonian_terms]

    return SparsePauliOp(pauliList, coeffs)
