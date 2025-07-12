import json
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import Counter
from src.sdes import sdes_encrypt
from src.prior import gaussian_prior
from src.ansatz import create_improved_ansatz
from src.utils import random_10bit_string, random_8bit_string, hamming_distance, to_jsonable
from src.optimization import (
    sdes_pretrain_ansatz_with_fidelity,
    simulate_final_circuit,
    optimize_vqa_scipy
)
from src.hamiltonian import build_hamiltonian
import config
from qiskit_ibm_runtime import QiskitRuntimeService, ibm_backend
import time


def main():
    start_time = time.time()
    
    backend = None
    if config.SDES_OPTIMIZATION_BACKEND == "IBMQ":
        service = QiskitRuntimeService()
        backend = service.least_busy()

    # ─── 1. Optionally fix RNG for reproducibility ─────────────────────────────────
    if config.SDES_GLOBAL_RANDOM_SEED is not None:
        np.random.seed(config.SDES_GLOBAL_RANDOM_SEED)

    # ─── 2. Decide whether to load custom pairs from JSON or generate randomly ──
    if config.SDES_USE_CUSTOM_PAIRS:
        with open(config.SDES_CUSTOM_PAIRS_FILE, "r") as f:
            pairs = json.load(f)
        true_key = pairs["true_key"]
        plaintexts = pairs["plaintexts"]
        ciphertexts = pairs["ciphertexts"]
    else:
        true_key = random_10bit_string()
        # true_key = "0000010101" 
        plaintexts = [random_8bit_string() for _ in range(config.SDES_NUM_PLAINTEXTS)]
        ciphertexts = [sdes_encrypt(pt, true_key) for pt in plaintexts]

    remarks = f"Prior key is {hamming_distance(true_key, config.SDES_PRIOR_CENTER_KEY)} bits away and a {config.SDES_GAUSSIAN_TYPE} gaussian with std dev {config.SDES_PRETRAIN_STD_DEV}  is used for pretraining with {config.SDES_PRETRAIN_OPTIMIZER} optimizer, cost function is {config.SDES_VQA_COST_FUNCTION} expectation value and vqa optimizer runs {config.SDES_VQA_OPTIMIZER} at {config.SDES_VQA_MAX_ITERATIONS} iters, {config.SDES_NUM_PLAINTEXTS} pairs of plain-cipher texts"
    print(f"True key:       {true_key}")
    # print(f"Plaintexts:     {plaintexts}")
    # print(f"Ciphertexts:    {ciphertexts}")

    # ─── 3. Build Gaussian prior (centered on either a guess or actual key) ──────
    if config.SDES_PRIOR_CENTER_KEY is None:
        center_key_str = true_key
    else:
        center_key_str = config.SDES_PRIOR_CENTER_KEY


    # ─── 4. Initialize the “U + CX” ansatz circuit ───────────────────────────────
    qc_template, theta_list, phi_list, lambda_list = create_improved_ansatz(
        num_qubits=config.SDES_NUM_QUBITS,
        reps=config.SDES_ANSATZ_REPS
    )
    flat_param_list = theta_list + phi_list + lambda_list

    # ─── 5. Pre-train ansatz to match Gaussian prior (with tqdm bar) ─────────────
    print("Pre-training ansatz to match Gaussian prior...")
    init_params, loss_history = sdes_pretrain_ansatz_with_fidelity(
    mean_bitstring=config.SDES_PRETRAIN_MEAN_BITSTRING,
    num_qubits=config.SDES_NUM_QUBITS,
    sigma=config.SDES_PRETRAIN_STD_DEV,
    reps=config.SDES_ANSATZ_REPS,
    maxiter=config.SDES_PRETRAIN_MAXITER,
    random_seed=config.SDES_PRETRAIN_SEED,
    optimizer=config.SDES_PRETRAIN_OPTIMIZER
    )
    with open("logs/pretrained_params.txt", "w") as f:
        f.write(str(init_params))
        
    with open("logs/pretrain_fidelity_loss.txt", "w") as f:
        f.write("\n".join(map(str, loss_history)))

        
    # ─── 6. Build Hamiltonian for each ciphertext ────────────────────────────────
    hamiltonians = [build_hamiltonian(ct) for ct in ciphertexts]

    # ─── 7. Run VQA optimization (shows tqdm bars) ────────────────────────────────
    print("Optimizing VQA for each plaintext/ciphertext pair...")
   
    vqa_results = optimize_vqa_scipy(
    hamiltonian_list=hamiltonians,
    qc_template=qc_template,
    param_list=flat_param_list,
    init_params=init_params,
    plaintexts=plaintexts,
    ciphertexts=ciphertexts,
    maxiter=config.SDES_VQA_MAX_ITERATIONS,
    tol=config.SDES_VQA_TOL,
    objective_function=config.SDES_VQA_COST_FUNCTION,
    optimizer=config.SDES_VQA_OPTIMIZER,
    backend=backend
    )
    

    # ─── 8. Voting over the rounds (sample final ansatz) ──────────────────────────
    bitstring_counter = Counter()
    final_qc_start_time = time.time()
    for k, best_params in enumerate(vqa_results["best_params_per_pair"]):
        counts = simulate_final_circuit(
            qc_template=qc_template,
            param_list=flat_param_list,
            params=best_params,
            ciphertext_plaintexts=plaintexts,
            rounds_key_idx=k,
            shots=config.SDES_NUM_SHOTS,
            backend=backend
        )
        for key_str, freq in counts.items():
            bitstring_counter[key_str] += freq
    final_qc_end_time = time.time()
    final_qc_elapsed_time = final_qc_end_time - final_qc_start_time
    with open("all_bitstring_frequencies.json", "w") as f:
        json.dump(dict(bitstring_counter), f, indent=2)

    guessed_key, total_count = bitstring_counter.most_common(1)[0]
    print(f"True key : {true_key}")
    print(f"Most frequent guessed key: {guessed_key} (count: {total_count})")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    # ─── 9. Save results to JSON ─────────────────────────────────────────────────
    output = {
        "remarks": remarks,
        "true_key": true_key,
        "prior_center_key": center_key_str,
        "prior_hamming_distance": hamming_distance(true_key, center_key_str),
        "guessed_key": guessed_key,
        "final_hamming_dist": hamming_distance(true_key, guessed_key),
        "guessed_key_count": total_count,
        "ansatz_reps": config.SDES_ANSATZ_REPS,
        "cost_history_per_pair" : vqa_results["cost_history_per_pair"],
        "top_10_bitstrings": bitstring_counter.most_common(10),
        "execution_time": f"{elapsed_time:.4f} seconds",
        "final_qc_execution_time": f"{final_qc_elapsed_time:.4f} seconds"
    }
    with open("runtime_comparison_results.json", "a") as f:
        json.dump(output, f, indent=2)

    print("Results saved to runtime_comparison_results.json")


if __name__ == "__main__":
    main()

