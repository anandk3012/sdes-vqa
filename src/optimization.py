import numpy as np
from scipy.optimize import minimize
from collections import Counter
from tqdm import tqdm, trange

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_ibm_runtime import Sampler, QiskitRuntimeService, ibm_backend, Estimator

from qiskit_aer import AerSimulator
from qiskit_finance.circuit.library import NormalDistribution
from hamming_gaussian import HammingGaussianDistribution

from src.prior import hamming_distance
from src.ansatz import create_improved_ansatz
from src.utils import encrypt_counts_keys, expectation_from_probabilities, expectation_from_estimator
from src.hamiltonian import build_hamiltonian
import time

# Pretrain Loss function
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

def sample_based_fidelity_loss(
    backend: ibm_backend.IBMBackend,
    target_probs: dict,
    qc: QuantumCircuit,
    param_list: list,
    params: np.ndarray,
    shots: int = 2048
) -> float:
    # Assign parameters
    binding = {p: v for p, v in zip(param_list, params)}
    # Add measurement (only needed for sampling)
    n = qc.num_qubits
    measured_qc = qc.copy()
    measured_qc.measure_all()
    transpiled_qc = transpile(measured_qc, backend=backend, optimization_level=3)

    # Run sampler
    sampler = Sampler(mode=backend)
    job = sampler.run([(transpiled_qc, binding)], shots=shots)
    result = job.result()._pub_results[0].data.c
    sampled_counts = result.get_counts()
      # key = int, value = prob

    # Convert target_probs (numpy array) to dict over int keys
    eps = 1e-9
    target_probs_clipped = np.clip(list(target_probs.values()), eps, 1)
    sampled_probs_clipped = np.array([sampled_counts.get(i, 0) for i in range(2**n)])
    sampled_probs_clipped = np.clip(sampled_probs_clipped, eps, 1)

    # Compute classical fidelity: F = (sum sqrt(p_i q_i))^2
    overlap = np.sum(np.sqrt(target_probs_clipped * sampled_probs_clipped))
    fidelity = overlap ** 2
    return 1 - fidelity

# Pre training params
def saes_pretrain_ansatz_with_fidelity(
    mean_bitstring: str,
    num_qubits: int = 10,
    sigma: float = 2.5,
    reps: int = 5,
    maxiter: int = 300,
    random_seed: int = 42,
    optimizer: str = "COBYLA",
    backend : ibm_backend.IBMBackend = None,
    shots: int = 2048
):
    print(f"Std_dev for Gaussian prior: {sigma}")
    np.random.seed(random_seed)
    mu = int(mean_bitstring, 2)
    bounds = (0, 2**num_qubits - 1)

    # sv_init_time = time.time()
    # hamming_init = NormalDistribution(num_qubits=num_qubits, mu=mu, sigma=sigma, bounds=bounds)
    # target_probs = Statevector.from_instruction(hamming_init.decompose()).probabilities_dict()
    # sv_init_end_time = time.time()
    # sv_init_elapsed_time = sv_init_end_time - sv_init_time
    # print(f"Statevector initialization took {sv_init_elapsed_time:.2f} seconds.")
    ansatz, θ_list, φ_list, λ_list = create_improved_ansatz(num_qubits, reps)
    param_list = θ_list + φ_list + λ_list
    init_params = np.random.uniform(0, 2 * np.pi, len(param_list))
    return init_params, []
    # assert len(param_list) == len(init_params), "Mismatch between circuit parameters and initial parameter values!"
    
    # loss_history = []
    # pbar = tqdm(total=maxiter, desc="Pretraining Fidelity Loss")

    # if num_qubits <= 10:
    #     # Use statevector fidelity
    #     def loss_fn(params):
    #         sv_ansatz = Statevector.from_instruction(
    #             ansatz.assign_parameters({p: v for p, v in zip(param_list, params)})
    #         ).data
    #         sv_target = Statevector.from_instruction(hamming_init.decompose()).data
    #         fidelity = np.abs(np.vdot(sv_target, sv_ansatz))**2
    #         loss = 1 - fidelity
    #         loss_history.append(loss)
    #         return loss

    # else:
    #     def loss_fn(params):
    #         loss = sample_based_fidelity_loss(backend, target_probs, ansatz, param_list, params, shots=shots)
    #         loss_history.append(loss)
    #         return loss

    # def callback(params):
    #     pbar.update(1)

    # initial_loss = loss_fn(init_params)
    # print(f"Initial Fidelity Loss: {initial_loss:.6f}")

    # result = minimize(
    #     loss_fn,
    #     init_params,
    #     method=optimizer,
    #     options={"maxiter": maxiter, "disp": False},
    #     callback=callback
    # )

    # pbar.close()
    # final_loss = loss_fn(result.x)
    # print("Pretraining completed.")
    # print(f"Initial Fidelity Loss: {initial_loss:.6f}, Final Fidelity Loss: {final_loss:.6f}")

    # return result.x, loss_history



# Pre training params
def sdes_pretrain_ansatz_with_fidelity(
    mean_bitstring: str,
    num_qubits: int = 10,
    sigma: float = 2.5,
    reps: int = 5,
    maxiter: int = 300,
    random_seed: int = 42,
    optimizer: str = "COBYLA"
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
    np.random.seed(random_seed)
    mu = int(mean_bitstring, 2)
    bounds = (0, 2**num_qubits - 1)

    hamming_init = NormalDistribution(num_qubits=num_qubits, mu=mu, sigma=sigma, bounds=bounds)
    # hamming_init = HammingGaussianDistribution(num_qubits, mean_bitstring, sigma)
    
    target_statevector = Statevector.from_instruction(hamming_init.decompose()).data


    ansatz, θ_list, φ_list, λ_list = create_improved_ansatz(num_qubits, reps)
    param_list = θ_list + φ_list + λ_list
    init_params = np.random.uniform(0, 2 * np.pi, len(param_list))

    assert len(param_list) == len(init_params), "Mismatch between circuit parameters and initial parameter values!"
    pbar = tqdm(total=maxiter, desc="Pretraining Fidelity Loss")
    loss_history = []
    """    Callback function to update the progress bar and record loss history.
    This is called by the optimizer
    after each iteration.
    """
    def callback(params):
        if len(loss_history) < maxiter:
            loss = fidelity_loss(params, target_statevector, ansatz, param_list)
            loss_history.append(loss)
            pbar.update(1)

    initial_loss = fidelity_loss(init_params, target_statevector, ansatz, param_list)
    print(f"Initial Fidelity Loss: {initial_loss:.6f}")
    result = minimize(
        lambda params: fidelity_loss(params, target_statevector, ansatz, param_list),
        init_params,
        method=optimizer,
        options={"maxiter": maxiter, "disp": False},
        callback=callback
    )
    pbar.close()
    print("Pretraining completed.")
    final_loss = fidelity_loss(result.x, target_statevector, ansatz, param_list)
    print(f"Initial Fidelity Loss: {initial_loss:.6f}, Final Fidelity Loss: {final_loss:.6f}")

    # Optionally save loss history
    # np.save("fidelity_loss_history.npy", np.array(loss_history))

    return result.x, loss_history


# Final Circuit Simulation
def simulate_final_circuit(
    qc_template: QuantumCircuit,
    param_list: list,
    params: np.ndarray,
    ciphertext_plaintexts: list,
    rounds_key_idx: int,
    shots: int = 1024,
    backend: ibm_backend.IBMBackend = None
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
    if backend is None:
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
    hamiltonian : SparsePauliOp,
    backend: ibm_backend.IBMBackend = None,
    shots: int = 1024
):
    # Bind parameters to the quantum circuit
    if backend is not None:
        qc_template.measure(range(qc_template.num_qubits), range(qc_template.num_qubits))
        transpiled_qc = transpile(qc_template, backend=backend)
        sampler = Sampler(mode=backend)
        binding = {p: v for p, v in zip(param_list, current_params)}
        job = sampler.run([(transpiled_qc, binding)], shots=shots)
        result = job.result()._pub_results[0].data.c
        counts = result.get_counts()
        # Convert counts to probabilities
        total_counts = sum(counts.values())
        if total_counts == 0:
            return 0.0
        probabilities = {k: v / total_counts for k, v in counts.items()}
        # Convert probabilities to counts
        counts = Counter({
            k: v for k, v in probabilities.items()
        })
    else:
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
    optimizer : str = "COBYLA",
    backend : ibm_backend.IBMBackend = None
):
    

    def objective_function_wrapper(current_params, qc_template, param_list, plaintext, hamiltonian, backend):
        return default_objective_function(current_params, qc_template, param_list, plaintext, hamiltonian, backend=backend)
            

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
            cost = objective_function_wrapper(current_params, qc_template, param_list, plaintext, hamiltonian, backend=backend)
            cost_history.append(cost)
            return cost
        
        def callback(xk):
            pbar.update(1)

        result = minimize(
            scipy_objective,
            params,
            method=optimizer,
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
    optimizer: str = "COBYLA",
    backend: ibm_backend.IBMBackend = None
):
    def objective_function_wrapper(current_params, qc_template, param_list, plaintext, ciphertext, backend):
        return hamming_distance_objective(current_params, qc_template, param_list, plaintext, ciphertext, backend=backend)

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
            method=optimizer,
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
    objective_function : str = "hamiltonian",
    optimizer: str = "COBYLA",
    backend: ibm_backend.IBMBackend = None
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
        return optimize_vqa_hamming(qc_template, param_list, init_params, plaintexts, ciphertexts, maxiter, tol,
                                      optimizer=optimizer, backend=backend)

    return optimize_vqa_hamiltonian(hamiltonian_list, qc_template, param_list, init_params, plaintexts, maxiter, tol,
                                      optimizer=optimizer, backend=backend)



