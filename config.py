# config.py
import os



# SDES Configurations
# ─── Problem Definition ──────────────────────────────────────────────────────
SDES_OPTIMIZATION_BACKEND  = "IBMQ"  # "IBMQ" or "local"

# How many random plaintext/ciphertext pairs should we generate? 
# (if you want to supply your own pairs, set USE_CUSTOM_PAIRS = True below)
SDES_NUM_PLAINTEXTS = 1    
SDES_USE_CUSTOM_PAIRS = False      

# If USE_CUSTOM_PAIRS=True, load from a JSON file (e.g. data/example_keys.json).
# Otherwise we randomly sample plaintexts and let main.py encrypt with a random key.
SDES_CUSTOM_PAIRS_FILE = "data/example_keys.json"


# ─── Gaussian Prior ────────────────────────────────────────────────────────────

# If None, we center the Gaussian prior on the true key. 
# Otherwise set this to a 10-bit string to center on a guess.
SDES_PRIOR_CENTER_KEY = "0110010100"   #0111010010     
SDES_BITS_AWAY = "3 bits away"
SDES_GAUSSIAN_TYPE = "Normal"
# ─── Pretraining parameters ────────────────────────────────────
SDES_PRETRAIN_MEAN_BITSTRING = SDES_PRIOR_CENTER_KEY
SDES_PRETRAIN_STD_DEV = 4  # Standard deviation for Gaussian prior (how "wide" it is)

# Maximum iterations
SDES_PRETRAIN_MAXITER = 500
SDES_PRETRAIN_OPTIMIZER = "L-BFGS-B"

# Random seed for initializing ansatz angles before pretraining
SDES_PRETRAIN_SEED = 42              


# ─── Ansatz Configuration ──────────────────────────────────────────────────────

# How many qubits does our ansatz use?  (S-DES has a 10-bit key, so 10 qubits.)
SDES_NUM_QUBITS = 10                
SDES_ANSATZ_REPS = 5

# ─── VQA (Hamiltonian Optimization) Hyperparameters ───────────────────────────

SDES_VQA_NUM_EPOCHS = 4           # how many passes over max_iterations
SDES_VQA_MAX_ITERATIONS = 300     
SDES_VQA_LR = 0.3                     # learning rate for momentum-SGD
SDES_VQA_BETA = 0.9                   # momentum coefficient
SDES_VQA_TOL = 1e-6                   # grad-norm tolerance for early stopping
SDES_VQA_OPTIMIZER = "COBYLA"

SDES_VQA_COST_FUNCTION = "hamiltonian"
# ─── Sampling / Final Vote ─────────────────────────────────────────────────────

# Number of shots when sampling the final “best” ansatz on each round
SDES_NUM_SHOTS = 1024                 


# ─── Misc ──────────────────────────────────────────────────────────────────────

# If you want to fix numpy’s RNG (for total determinism), set this to an int.
SDES_GLOBAL_RANDOM_SEED = 12345  














# SAES Configurations
# ─── Problem Definition ──────────────────────────────────────────────────────
SAES_OPTIMIZATION_BACKEND  = "IBMQ"  # "IBMQ" or "local"

# How many random plaintext/ciphertext pairs should we generate? 
# (if you want to supply your own pairs, set USE_CUSTOM_PAIRS = True below)
SAES_NUM_PLAINTEXTS = 5    
SAES_USE_CUSTOM_PAIRS = False      

# If USE_CUSTOM_PAIRS=True, load from a JSON file (e.g. data/example_keys.json).
# Otherwise we randomly sample plaintexts and let main.py encrypt with a random key.
SAES_CUSTOM_PAIRS_FILE = "data/example_keys.json"


# ─── Gaussian Prior ────────────────────────────────────────────────────────────

# If None, we center the Gaussian prior on the true key. 
# Otherwise set this to a 16-bit string to center on a guess.
SAES_PRIOR_CENTER_KEY = "0000000000000000"      
SAES_BITS_AWAY = ""
SAES_GAUSSIAN_TYPE = "Normal"
# ─── Pretraining parameters ────────────────────────────────────
SAES_PRETRAIN_MEAN_BITSTRING = SAES_PRIOR_CENTER_KEY
SAES_PRETRAIN_STD_DEV = 8  # Standard deviation for Gaussian prior (how "wide" it is)

# Maximum iterations
SAES_PRETRAIN_MAXITER = 1
SAES_PRETRAIN_SHOTS = 1024
SAES_PRETRAIN_OPTIMIZER = "L-BFGS-B"

# Random seed for initializing ansatz angles before pretraining
SAES_PRETRAIN_SEED = 42              


# ─── Ansatz Configuration ──────────────────────────────────────────────────────

# How many qubits does our ansatz use?  (S-DES has a 10-bit key, so 10 qubits.)
SAES_NUM_QUBITS = 16                
SAES_ANSATZ_REPS = 5

# ─── VQA (Hamiltonian Optimization) Hyperparameters ───────────────────────────

SAES_VQA_NUM_EPOCHS = 1           # how many passes over max_iterations
SAES_VQA_MAX_ITERATIONS = 2     
SAES_VQA_LR = 0.3                     # learning rate for momentum-SGD
SAES_VQA_BETA = 0.9                   # momentum coefficient
SAES_VQA_TOL = 1e-6                   # grad-norm tolerance for early stopping
SAES_VQA_OPTIMIZER = "COBYLA"

SAES_VQA_COST_FUNCTION = "hamiltonian"
# ─── Sampling / Final Vote ─────────────────────────────────────────────────────

# Number of shots when sampling the final “best” ansatz on each round
SAES_NUM_SHOTS = 1024                 


# ─── Misc ──────────────────────────────────────────────────────────────────────

# If you want to fix numpy’s RNG (for total determinism), set this to an int.
SAES_GLOBAL_RANDOM_SEED = 12345  


















# API Keys
IBM_QUANTUM_API_KEY = os.getenv("IBM_QUANTUM_API_KEY")
IBM_CRN_INSTANCE_KEY = os.getenv("IBM_CRN_INSTANCE_KEY")