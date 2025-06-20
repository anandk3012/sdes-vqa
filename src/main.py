import json
import numpy as np
import os 
from collections import Counter
from src.sdes import sdes_encrypt
from src.prior import gaussian_prior
from src.ansatz import create_improved_ansatz
from src.utils import random_10bit_string, random_8bit_string, hamming_distance, to_jsonable
from src.optimization import (
    pretrain_ansatz_with_fidelity,
    simulate_final_circuit,
    optimize_vqa_scipy
)
from src.hamiltonian import build_hamiltonian
import config

def main():
    remarks = f"Prior key is {config.BITS_AWAY} and a {config.GAUSSIAN_TYPE} gaussian with std dev {config.PRETRAIN_STD_DEV}  is used for pretraining with {config.PRETRAIN_OPTIMIZER} optimizer, cost function is {config.VQA_COST_FUNCTION} expectation value and vqa optimizer runs {config.VQA_OPTIMIZER} at {config.VQA_MAX_ITERATIONS} iters, {config.NUM_PLAINTEXTS} pairs of plain-cipher texts"
    # ─── 1. Optionalnput("Remarks: ")ly fix RNG for reproducibility ─────────────────────────────────
    if config.GLOBAL_RANDOM_SEED is not None:
        np.random.seed(config.GLOBAL_RANDOM_SEED)

    # ─── 2. Decide whether to load custom pairs from JSON or generate randomly ──
    if config.USE_CUSTOM_PAIRS:
        with open(config.CUSTOM_PAIRS_FILE, "r") as f:
            pairs = json.load(f)
        true_key = pairs["true_key"]
        plaintexts = pairs["plaintexts"]
        ciphertexts = pairs["ciphertexts"]
    else:
        # true_key = random_10bit_string()
        true_key = "1011010101"
        plaintexts = [random_8bit_string() for _ in range(config.NUM_PLAINTEXTS)]
        ciphertexts = [sdes_encrypt(pt, true_key) for pt in plaintexts]

    print(f"True key:       {true_key}")
    # print(f"Plaintexts:     {plaintexts}")
    # print(f"Ciphertexts:    {ciphertexts}")

    # ─── 3. Build Gaussian prior (centered on either a guess or actual key) ──────
    if config.PRIOR_CENTER_KEY is None:
        center_key_str = true_key
    else:
        center_key_str = config.PRIOR_CENTER_KEY


    # ─── 4. Initialize the “U + CX” ansatz circuit ───────────────────────────────
    qc_template, theta_list, phi_list, lambda_list = create_improved_ansatz(
        num_qubits=config.NUM_QUBITS
    )
    flat_param_list = theta_list + phi_list + lambda_list

    # ─── 5. Pre-train ansatz to match Gaussian prior (with tqdm bar) ─────────────
    print("Pre-training ansatz to match Gaussian prior...")
    init_params, loss_history = pretrain_ansatz_with_fidelity(
    mean_bitstring=config.PRETRAIN_MEAN_BITSTRING,
    num_qubits=config.NUM_QUBITS,
    sigma=config.PRETRAIN_STD_DEV,
    reps=config.ANSATZ_REPS,
    maxiter=config.PRETRAIN_MAXITER,
    random_seed=config.PRETRAIN_SEED
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
    ciphertexts = ciphertexts,
    maxiter=config.VQA_MAX_ITERATIONS,
    tol=config.VQA_TOL,
    objective_function=config.VQA_COST_FUNCTION  
    )
    

    # ─── 8. Voting over the rounds (sample final ansatz) ──────────────────────────
    bitstring_counter = Counter()

    for k, best_params in enumerate(vqa_results["best_params_per_pair"]):
        counts = simulate_final_circuit(
            qc_template=qc_template,
            param_list=flat_param_list,
            params=best_params,
            ciphertext_plaintexts=plaintexts,
            rounds_key_idx=k,
            shots=config.NUM_SHOTS
        )
        for key_str, freq in counts.items():

            bitstring_counter[key_str] += freq

    with open("all_bitstring_frequencies.json", "w") as f:
        json.dump(dict(bitstring_counter), f, indent=2)

    guessed_key, total_count = bitstring_counter.most_common(1)[0]
    print(f"True key : {true_key}")
    print(f"Most frequent guessed key: {guessed_key} (count: {total_count})")

    # ─── 9. Save results to JSON ─────────────────────────────────────────────────
    output = {
        "remarks": remarks,
        "true_key": true_key,
        "prior_center_key": center_key_str,
        "guessed_key": guessed_key,
        "guessed_key_count": total_count,
        "hamming_dist": hamming_distance(true_key, guessed_key),
        "cost_history_per_pair" : vqa_results["cost_history_per_pair"],
        "top_20_bitstrings": bitstring_counter.most_common(20)
    }
    with open("results2.json", "a") as f:
        json.dump(output, f, indent=2)

    print("Results saved to results2.json")


if __name__ == "__main__":
    main()

