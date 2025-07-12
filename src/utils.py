import numpy as np
from collections import Counter
from src.sdes import sdes_encrypt, s_aes_encrypt

from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_ibm_runtime import EstimatorV2 as Estimator, ibm_backend
from qiskit import QuantumCircuit

def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    return obj

def hamming_distance(a: str, b: str) -> int:
    """
    Compute the Hamming distance between two bit strings of equal length.
    Args:
        a: First bit string (e.g., '0101')
        b: Second bit string (e.g., '1101')
    Returns:
        Number of positions at which the corresponding bits are different.
    """
    if len(a) != len(b):
        raise ValueError("Bit strings must be of equal length.")
    return sum(x != y for x, y in zip(a, b))

def copy_circ(gate_list, num_qubits):
    """
    Build a small QuantumCircuit applying CNOTs in sequence.
    gate_list: list of (control, target) tuples.
    num_qubits: total number of qubits.
    Returns:
        QuantumCircuit implementing those CNOTs.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_qubits)
    for control, target in gate_list:
        qc.cx(control, target)
    return qc

def swap_gates():
    """
    Build an 8-qubit SWAP network swapping pairs (0↔4, 1↔5, 2↔6, 3↔7).
    Returns:
        QuantumCircuit on 8 qubits implementing those 4 SWAPs.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(8)
    pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in pairs:
        qc.swap(a, b)
    return qc

def get_int_from_binary(bit_list):
    """
    Convert a list of bits (0/1) to an integer.
    Example: [1,0,1] → 5.
    """
    return int("".join(str(b) for b in bit_list), 2)

def random_bitstring(length: int) -> str:
    """
    Return a random bitstring of given length (characters '0' or '1').
    """
    return "".join(np.random.choice(['0', '1'], size=length))

def random_10bit_string() -> str:
    """Return a random 10-bit string."""
    return random_bitstring(10)

def random_16bit_string() -> str:
    """Return a random 16-bit string."""
    return random_bitstring(16)

def random_8bit_string() -> str:
    """Return a random 8-bit string."""
    return random_bitstring(8)

def encrypt_counts_keys(counts: dict, plaintext: str) -> Counter:
    """
    Given a counts dictionary mapping 10-bit key strings → frequencies,
    run sdes_encrypt(plaintext, key_str) to obtain ciphertext (8-bit),
    then tally frequencies of those ciphertexts.

    Args:
        counts: dict { '0101010101': freq, ... } representing key → freq
        plaintext: 8-bit string used to encrypt with each key_str

    Returns:
        Counter { 'ciphertext8': aggregated_freq, ... }
    """
    result = Counter()
    for key_str, freq in counts.items():
        if(len(key_str) == 10):
            cipher = sdes_encrypt(plaintext, key_str)
        elif (len(key_str) == 16):
            cipher = s_aes_encrypt(plaintext, key_str[:10])
        result[cipher] += freq
    return result

def expectation_from_probabilities(probs: dict, hamiltonian: SparsePauliOp) -> float:
    """
    Compute expectation ⟨H⟩ given a probability distribution over 8-bit strings.
    Args:
        probs: dict { '01010101': probability, ... } over 8-bit outcomes.
        hamiltonian: SparsePauliOp acting on 8 qubits.

    Returns:
        Scalar ⟨H⟩ = sum_x Pr(x) * H(x), where H(x) is the diagonal eigenvalue.
    """
    def basis_state_expectation(cipher_text : str, hamiltonian : SparsePauliOp):
        sv = Statevector.from_label(cipher_text)
        exp = sv.expectation_value(hamiltonian)
        return exp    
    
    expectation_value = (np.real(list([(basis_state_expectation(k, hamiltonian) * v) for k, v in probs.items()])))
    exp_val = expectation_value.sum()
    
    return exp_val

def expectation_from_estimator(hamiltonian: SparsePauliOp, qc: QuantumCircuit, params, backend : ibm_backend.IBMBackend = None ,shots: int = 1024) -> Estimator:
    """
    Estimate the expectation value of a Hamiltonian using a quantum circuit.
    Args:
        hamiltonian: SparsePauliOp representing the Hamiltonian.
        qc: QuantumCircuit to be used for sampling.
        shots: Number of shots for sampling.

    Returns:
        Scalar expectation value ⟨H⟩.
    """
    if backend is None:
        backend = AerSimulator()
    estimator = Estimator(mode=backend)
    job = estimator.run([(qc, hamiltonian, params)])
    result = job.result()
    exp_value = float(result[0].data.evs)
    print(exp_value)

    return exp_value