import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp


def build_hamiltonian(ciphertext_str: str) -> SparsePauliOp:
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
