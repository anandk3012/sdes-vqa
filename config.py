# config.py

# ─── Problem Definition ──────────────────────────────────────────────────────

# How many random plaintext/ciphertext pairs should we generate? 
# (if you want to supply your own pairs, set USE_CUSTOM_PAIRS = True below)
NUM_PLAINTEXTS = 30            
USE_CUSTOM_PAIRS = False      

# If USE_CUSTOM_PAIRS=True, load from a JSON file (e.g. data/example_keys.json).
# Otherwise we randomly sample plaintexts and let main.py encrypt with a random key.
CUSTOM_PAIRS_FILE = "data/example_keys.json"


# ─── Gaussian Prior ────────────────────────────────────────────────────────────

# If None, we center the Gaussian prior on the true key. 
# Otherwise set this to a 10-bit string to center on a guess.
PRIOR_CENTER_KEY = "0101010101"        
BITS_AWAY = "completely random"
GAUSSIAN_TYPE = "Normal"
# ─── Pretraining parameters ────────────────────────────────────
PRETRAIN_MEAN_BITSTRING = PRIOR_CENTER_KEY
PRETRAIN_STD_DEV = 6.5

# Maximum iterations
PRETRAIN_MAXITER = 500   
PRETRAIN_OPTIMIZER = "L-BFGS-B"      

# Random seed for initializing ansatz angles before pretraining
PRETRAIN_SEED = 42              
   


# ─── Ansatz Configuration ──────────────────────────────────────────────────────

# How many qubits does our ansatz use?  (S-DES has a 10-bit key, so 10 qubits.)
NUM_QUBITS = 10                
ANSATZ_REPS = 5

# ─── VQA (Hamiltonian Optimization) Hyperparameters ───────────────────────────

VQA_NUM_EPOCHS = 4           # how many passes over max_iterations
VQA_MAX_ITERATIONS = 300     
VQA_LR = 0.3                     # learning rate for momentum-SGD
VQA_BETA = 0.9                   # momentum coefficient
VQA_TOL = 1e-6                   # grad-norm tolerance for early stopping
VQA_OPTIMIZER = "COBYLA"

VQA_COST_FUNCTION = "hamiltonian"
# ─── Sampling / Final Vote ─────────────────────────────────────────────────────

# Number of shots when sampling the final “best” ansatz on each round
NUM_SHOTS = 1024                 


# ─── Misc ──────────────────────────────────────────────────────────────────────

# If you want to fix numpy’s RNG (for total determinism), set this to an int.
GLOBAL_RANDOM_SEED = None     