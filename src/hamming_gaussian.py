import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import Initialize

class HammingGaussianDistribution(QuantumCircuit):
    """
    Encodes a discrete Gaussian over Hamming distance to `mean_bitstring`.

    The probability of each basis state |x> is
        p(x) ∝ exp(-HD(x,mean)^2 / (2σ^2))
    and we load √p(x) into amplitudes so sampling matches p(x).

    Args:
        num_qubits: number of qubits (must match len(mean_bitstring))
        mean_bitstring: center bitstring (e.g. "01101")
        sigma: standard deviation for Hamming-distance Gaussian
        name: circuit name
    """
    def __init__(
        self,
        num_qubits: int,
        mean_bitstring: str,
        sigma: float,
        name: str = "P_Hamming"
    ):
        #— sanity checks
        if len(mean_bitstring) != num_qubits:
            raise ValueError("len(mean_bitstring) must == num_qubits")
        super().__init__(num_qubits, name=name)

        #— build classical probability vector
        center = np.array([int(b) for b in mean_bitstring])
        N = 2**num_qubits

        # compute Hamming distance for each integer 0..N-1
        dists = np.zeros(N, dtype=int)
        for i in range(N):
            bits = np.array(list(map(int, format(i, f'0{num_qubits}b'))))
            dists[i] = np.count_nonzero(bits != center)

        probs = np.exp(-0.5 * (dists / sigma)**2)
        probs /= probs.sum()

        #— prepare amplitudes and circuit
        amps = np.sqrt(probs)
        initializer = Initialize(amps)
        prep = initializer.gates_to_uncompute().inverse()

        # compose into this circuit
        self.append(prep, self.qubits)

        #— store for introspection
        self._hamming = dists
        self._probabilities = probs

    @property
    def hamming_distances(self) -> np.ndarray:
        """Array of HD(x,center) for x=0..2^n-1."""
        return self._hamming

    @property
    def probabilities(self) -> np.ndarray:
        """Sampling probabilities over basis states."""
        return self._probabilities
