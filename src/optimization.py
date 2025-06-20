import numpy as np
from scipy.optimize import minimize
from collections import Counter
from tqdm import tqdm, trange

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_ibm_runtime import Sampler
from qiskit_aer import AerSimulator
from qiskit_finance.circuit.library import NormalDistribution
from hamming_gaussian import HammingGaussianDistribution

from src.prior import hamming_distance
from src.ansatz import create_improved_ansatz
from src.utils import encrypt_counts_keys, expectation_from_probabilities, expectation_from_estimator
from src.hamiltonian import build_hamiltonian


def get_probabilities_statevector(
    qc: QuantumCircuit,
    param_list: list,
    numeric_params: np.ndarray
) -> np.ndarray:
    """
    Bind numeric_params to param_list in qc, simulate statevector,
    and return a length-(2^n) array of probabilities.
    """
    bound_qc = qc.assign_parameters(
        {p: v for p, v in zip(param_list, numeric_params)},
        inplace=False
    )
    sv = Statevector.from_instruction(bound_qc)
    return sv.probabilities()

# Loss functions

def kl_divergence_loss(
    numeric_params: np.ndarray,
    prior_probs: np.ndarray,
    qc: QuantumCircuit,
    param_list: list
) -> float:
    """
    Compute KL(prior || ansatz_output) = sum_i prior[i] * log(prior[i]/q[i]),
    where q is the distribution produced by the ansatz statevector.
    """
    q_probs = get_probabilities_statevector(qc, param_list, numeric_params)
    eps = 1e-10
    q_probs = np.clip(q_probs, eps, 1.0)
    prior_clipped = np.clip(prior_probs, eps, 1.0)
    return float(np.sum(prior_clipped * np.log(prior_clipped / q_probs)))

def fidelity_loss(
    numeric_params: np.ndarray,
    target_sv: np.ndarray,
    qc: QuantumCircuit,
    param_list: list
) -> float:
    """
    1 - fidelity between the target statevector and the ansatz-generated state.
    """
    sv_ansatz = Statevector.from_instruction(
        qc.assign_parameters({p: v for p, v in zip(param_list, numeric_params)})
    ).data

    fidelity = np.abs(np.vdot(target_sv, sv_ansatz))**2
    return 1 - fidelity


# Pre training params

def pretrain_ansatz(
    qc: QuantumCircuit,
    param_list: list,
    prior_probs: np.ndarray,
    maxiter: int = 300,
    random_seed: int = 42
) -> np.ndarray:
    """
    Run L-BFGS-B to minimize KL divergence between a Gaussian prior and ansatz output.
    Shows a tqdm bar for each L-BFGS-B iteration (up to maxiter).

    Returns:
        The optimized flat parameter array (numpy.ndarray).
    """
    np.random.seed(random_seed)
    init_params = np.random.uniform(0, 2 * np.pi, len(param_list))

    # Create a tqdm progress bar for L-BFGS-B iterations
    pbar = tqdm(total=maxiter, desc="Pretrain KL Loss", leave=True)

    history = []
    
    def objective(params):
        kl = kl_divergence_loss(params, prior_probs, qc, param_list)
        history.append(kl)
        
        return kl
    
    
    def callback(params):
        # Each time the optimizer calls back, advance the bar by 1
        pbar.update(1)
        
    kl_init = kl_divergence_loss(init_params, prior_probs, qc, param_list)

    res = minimize(
        objective,
        init_params,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "disp": False},
        callback=callback
    )
    pbar.close()
    
    with open("kl_history.txt", "w") as f:
        f.write(str(np.array(history)))
    
    kl_final = kl_divergence_loss(res.x, prior_probs, qc, param_list)
    print(f"Initial KL: {kl_init:.6f}, Final KL: {kl_final:.6f}")
    
    return res.x  # optimized parameter vector

def pretrain_ansatz_with_fidelity(
    mean_bitstring: str,
    num_qubits: int = 10,
    sigma: float = 2.5,
    reps: int = 5,
    maxiter: int = 300,
    random_seed: int = 42
):
    """
    Pre-train the ansatz parameters to maximize fidelity with a target Gaussian state.

    Args:
        mean_bitstring: Center of the Gaussian as a bitstring.
        num_qubits: Number of qubits.
        sigma: Standard deviation for the Gaussian.
        reps: Number of ansatz repetitions.
        maxiter: Maximum optimizer iterations.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple (optimized_params, loss_history)
        
    """
    
    print("Std_dev for Gaussian prior:", sigma)
    print("\n")
    np.random.seed(random_seed)
    mu = int(mean_bitstring, 2)
    bounds = (0, 2**num_qubits - 1)

    hamming_init = NormalDistribution(num_qubits=num_qubits, mu=mu, sigma=sigma, bounds=bounds)
    # hamming_init = HammingGaussianDistribution(num_qubits, mean_bitstring, sigma)
    target_statevector = Statevector.from_instruction(hamming_init.decompose()).data


    ansatz, θ_list, φ_list, λ_list = create_improved_ansatz(num_qubits, reps)
    param_list = θ_list + φ_list + λ_list

    init_params = np.random.uniform(0, 2 * np.pi, len(param_list))
    pbar = tqdm(total=maxiter, desc="Pretraining Fidelity Loss")
    loss_history = []

    def callback(params):
        if len(loss_history) < maxiter:
            loss = fidelity_loss(params, target_statevector, ansatz, param_list)
            loss_history.append(loss)
            pbar.update(1)

    initial_loss = fidelity_loss(init_params, target_statevector, ansatz, param_list)

    result = minimize(
        lambda params: fidelity_loss(params, target_statevector, ansatz, param_list),
        init_params,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "disp": False},
        callback=callback
    )
    pbar.close()

    final_loss = fidelity_loss(result.x, target_statevector, ansatz, param_list)
    print(f"Initial Fidelity Loss: {initial_loss:.6f}, Final Fidelity Loss: {final_loss:.6f}")

    # Optionally save loss history
    # np.save("fidelity_loss_history.npy", np.array(loss_history))

    return result.x, loss_history

def compute_gradient_statevector(
    qc_template: QuantumCircuit,
    flat_params: np.ndarray,
    flat_param_list: list,
    hamiltonian,
    ciphertext_plaintext: str,
    shift: float = np.pi / 2
) -> np.ndarray:
    """
    Compute gradient of ⟨H⟩ w.r.t. each parameter in flat_params via parameter-shift rule,
    using statevector simulator (exact).
    Args:
        qc_template: QuantumCircuit with symbolic parameters (ansatz only, no measures).
        flat_params: numpy array of current parameter values.
        flat_param_list: list of Parameter symbols corresponding to flat_params.
        hamiltonian: SparsePauliOp for this ciphertext.
        ciphertext_plaintext: the 8-bit plaintext string used for encryption in this round.
        shift: parameter-shift amount (π/2 by default).
    Returns:
        grad: numpy array, same shape as flat_params.
    """
    num_params = len(flat_params)
    grad = np.zeros(num_params)

    for idx in range(num_params):
        plus = flat_params.copy()
        minus = flat_params.copy()
        plus[idx] += shift
        minus[idx] -= shift

        # Build circuits for shifted parameters using assign_parameters:
        qc_plus = qc_template.assign_parameters(
            {p: v for p, v in zip(flat_param_list, plus)},
            inplace=False
        )
        qc_minus = qc_template.assign_parameters(
            {p: v for p, v in zip(flat_param_list, minus)},
            inplace=False
        )

        # Simulate statevectors for shifted parameters
        probs_plus = Statevector.from_instruction(qc_plus).probabilities()
        probs_minus = Statevector.from_instruction(qc_minus).probabilities()

        # Build counts-like dictionaries: key_str → prob
        counts_plus = Counter({
            format(i, f"0{qc_template.num_qubits}b"): p
            for i, p in enumerate(probs_plus)
        })
        counts_minus = Counter({
            format(i, f"0{qc_template.num_qubits}b"): p
            for i, p in enumerate(probs_minus)
        })

        # Convert key-based “counts” to ciphertext-based counts
        cipher_plus = encrypt_counts_keys(counts_plus, ciphertext_plaintext)
        cipher_minus = encrypt_counts_keys(counts_minus, ciphertext_plaintext)

        total_plus = sum(cipher_plus.values())
        total_minus = sum(cipher_minus.values())

        # Convert to probabilities over 8-bit ciphertexts
        probs_cipher_plus = {c: freq / total_plus for c, freq in cipher_plus.items()}
        probs_cipher_minus = {c: freq / total_minus for c, freq in cipher_minus.items()}

        exp_plus = expectation_from_probabilities(probs_cipher_plus, hamiltonian)
        exp_minus = expectation_from_probabilities(probs_cipher_minus, hamiltonian)

        # Take the real part explicitly to avoid ComplexWarning
        grad[idx] = float(np.real(exp_plus - exp_minus)) * 0.5

    return grad



# Circuit sims
def simulate_final_circuit(
    qc_template: QuantumCircuit,
    param_list: list,
    params: np.ndarray,
    ciphertext_plaintexts: list,
    rounds_key_idx: int,
    shots: int = 1024
):
    """
    Bind params to qc_template, add measurements, run a Sampler job,
    collect counts over 10-bit keys, convert to frequencies, then return
    the top-3 most frequent keys (as a list of (key_str, freq) tuples).

    Args:
        qc_template: QuantumCircuit (ansatz without measurements).
        param_list: list of Parameter symbols.
        params: flat numpy array of values to bind.
        ciphertext_plaintexts: list of 5 plaintexts (8-bit strings).
        rounds_key_idx: index in ciphertext_plaintexts to pick which plaintext to use
                        when converting key→ciphertext for expectation.
        shots: number of shots for Sampler.

    Returns:
        topk: list of (key_str, frequency) for the 3 most frequent measured keys.
    """
    # Bind parameters and add measurements
    n = qc_template.num_qubits
    qc = QuantumCircuit(n,n)
    qc.compose(qc_template, inplace=True)
    qc.measure(range(n), range(n))

    # Transpile and sample on backend
    backend = AerSimulator(method='matrix_product_state')
    transpiled = transpile(qc, backend=backend)
    
    sampler = Sampler(mode=backend)
    
    binding = {p: v for p, v in zip(param_list, params)}
    job = sampler.run([(transpiled, binding)], shots=shots)
    
    result = job.result()._pub_results[0].data.c
    counts = result.get_counts()
    

    # Return top-3 keys
    return counts


# Objective Functions 
def default_objective_function(
    current_params,
    qc_template : QuantumCircuit,
    param_list : list,
    plaintext : str,
    hamiltonian : SparsePauliOp
):
    # Bind parameters to the quantum circuit
    bound_qc = qc_template.assign_parameters(
        {p: v for p, v in zip(param_list, current_params)},
        inplace=False
    )
    # Simulate the statevector
    statevector = Statevector.from_instruction(bound_qc)
    probabilities = statevector.probabilities()
    # Convert probabilities to counts
    counts = Counter({
        format(i, f"0{qc_template.num_qubits}b"): p
        for i, p in enumerate(probabilities)
    })
    # Encrypt counts based on the plaintext
    encrypted_counts = encrypt_counts_keys(counts, plaintext)
    # Normalize to get probabilities
    total = sum(encrypted_counts.values())
    encrypted_probs = {k: (v / total) for k, v in encrypted_counts.items()}
    # Calculate expectation value
    expectation = expectation_from_probabilities(encrypted_probs, hamiltonian)
    return np.real(expectation)

def hamming_distance_objective(
    current_params,
    qc_template,
    param_list,
    plaintext,
    actual_ciphertext
):
    """
    Objective: Expected Hamming distance between the ciphertexts produced by sampled keys
    and the actual ciphertext for the given plaintext.

    Lower is better (0 means all probability on the correct ciphertext).
    """
    # Bind parameters to the quantum circuit
    bound_qc = qc_template.assign_parameters(
        {p: v for p, v in zip(param_list, current_params)},
        inplace=False
    )
    # Simulate the statevector
    statevector = Statevector.from_instruction(bound_qc)
    probabilities = statevector.probabilities()
    # Convert probabilities to counts (over keys)
    counts = Counter({
        format(i, f"0{qc_template.num_qubits}b"): p
        for i, p in enumerate(probabilities)
    })
    # Map each key to its ciphertext
    from src.sdes import sdes_encrypt
    ciphertext_probs = Counter()
    for key, prob in counts.items():
        ciph = sdes_encrypt(plaintext, key)
        ciphertext_probs[ciph] += prob
    # Normalize
    total = sum(ciphertext_probs.values())
    if total == 0:
        return len(actual_ciphertext)  # maximal distance if no probability
    for c in ciphertext_probs:
        ciphertext_probs[c] /= total
    # Compute expected Hamming distance
    from src.utils import hamming_distance
    expected_hd = sum(
        p * hamming_distance(c, actual_ciphertext)
        for c, p in ciphertext_probs.items()
    )
    return expected_hd


# Optimization Functions
def optimize_vqa_hamiltonian(
    hamiltonian_list : list[SparsePauliOp],
    qc_template: QuantumCircuit,
    param_list: list,
    init_params: np.ndarray,
    plaintexts: list,
    maxiter: int = 500,
    tol: float = 1e-6,
):
    

    def objective_function_wrapper(current_params, qc_template, param_list, plaintext, hamiltonian):
        return default_objective_function(current_params, qc_template, param_list, plaintext, hamiltonian)
            

    results = {
        "best_params_per_pair": [],
        "cost_history_per_pair": []
    }

    params = init_params.copy()
    
    for k, hamiltonian in enumerate(hamiltonian_list):
        plaintext = plaintexts[k]
        cost_history = []

        pbar = tqdm(total=maxiter, desc=f"Optimizing pair {k+1}/{len(hamiltonian_list)}", leave=True)

        def scipy_objective(current_params):
            cost = objective_function_wrapper(current_params, qc_template, param_list, plaintext, hamiltonian)
            cost_history.append(cost)
            return cost
        
        def callback(xk):
            pbar.update(1)

        result = minimize(
            scipy_objective,
            params,
            method='COBYLA',
            options={'maxiter': maxiter, 'disp': False},
            callback=callback
        )

        pbar.close()
        params = result.x
        results["best_params_per_pair"].append(params.copy())
        results["cost_history_per_pair"].append(cost_history)

    return results

def optimize_vqa_hamming(
    qc_template: QuantumCircuit,
    param_list: list,
    init_params: np.ndarray,
    plaintexts: list,
    ciphertexts: list,
    maxiter: int = 500,
    tol: float = 1e-6,
):
    def objective_function_wrapper(current_params, qc_template, param_list, plaintext, ciphertext):
        return hamming_distance_objective(current_params, qc_template, param_list, plaintext, ciphertext)
    
    results = {
        "best_params_per_pair": [],
        "cost_history_per_pair": []
    }

    params = init_params.copy()
    
    for k, ciphertext in enumerate(ciphertexts):
        plaintext = plaintexts[k]
        cost_history = []

        pbar = tqdm(total=maxiter, desc=f"Optimizing pair {k+1}/{len(ciphertexts)}", leave=True)

        def scipy_objective(current_params):
            cost = objective_function_wrapper(current_params, qc_template, param_list, plaintext, ciphertext)
            cost_history.append(cost)
            return cost
        
        def callback(xk):
            pbar.update(1)

        result = minimize(
            scipy_objective,
            params,
            method='COBYLA',
            options={'maxiter': maxiter, 'disp': False},
            callback=callback
        )

        pbar.close()
        params = result.x
        results["best_params_per_pair"].append(params.copy())
        results["cost_history_per_pair"].append(cost_history)

    return results
    

def optimize_vqa_scipy(
    hamiltonian_list : list[SparsePauliOp],
    qc_template: QuantumCircuit,
    param_list: list,
    init_params: np.ndarray,
    plaintexts: list,
    ciphertexts: list,
    maxiter: int = 500,
    tol: float = 1e-6,
    objective_function : str = "hamiltonian"
):
    """
    Optimize the parameters of a quantum circuit using SciPy's minimize function.

    Args:
        hamiltonian_list: List of Hamiltonians corresponding to each ciphertext.
        qc_template: Parameterized quantum circuit template.
        param_list: List of parameters in the quantum circuit.
        init_params: Initial parameter values.
        ciphertext_plaintexts: List of plaintexts corresponding to each ciphertext.
        maxiter: Maximum number of iterations for the optimizer.
        tol: Tolerance for convergence.
        objective_function: Callable for the cost function. If None, uses default.

    Returns:
        Dictionary containing optimized parameters and cost history for each pair.
    """
    
    if objective_function == "hamming":
        return optimize_vqa_hamming(qc_template, param_list, init_params, plaintexts, ciphertexts, maxiter, tol) 

    return optimize_vqa_hamiltonian(hamiltonian_list, qc_template, param_list, init_params, plaintexts, maxiter, tol)



