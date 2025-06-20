# VQA-S-DES

**A Variational Quantum Algorithm for S-DES Key Recovery**

This repository contains an end-to-end pipeline that:
1. Implements the toy S-DES cipher in Python.
2. Defines a Gaussian prior over all 10-bit keys.
3. Builds several variational ansätze in Qiskit.
4. Pre-trains the ansatz to match the prior (KL divergence).
5. Constructs a Hamiltonian cost function from ciphertexts.
6. Runs a VQA loop (momentum-SGD + parameter-shift gradients).
7. “Votes” over multiple plaintext/ciphertext pairs to recover the key.

---

## 📦 Installation

## 1. Clone the Repo

```bash
git clone https://github.com/anandk3012/vqa-sdes.git
cd vqa-sdes
```
---

## 2. Inspect the File Structure

```
vqa-sdes/
├── config.py
├── README.md
├── requirements.txt
├── results.json
├── setup.py
├── src/
│   ├── __init__.py
│   ├── ansatz.py
│   ├── hamiltonian.py
│   ├── optimization.py
│   ├── prior.py
│   ├── sdes.py
│   ├── utils.py
│   └── main.py
├── tests/
│   ├── test_ansatz.py
│   ├── test_hamiltonian.py
│   ├── test_prior.py
│   ├── test_sdes.py
│   └── test_utils.py
└── data/
    └── example_keys.json
```

- `config.py` holds every hyperparameter and “knob” you can tweak.  
- `results.json` Stores the obtained key counts and results.  
- `src/` contains all core modules.  
- `tests/` contains pytest-compatible unit tests.  
- `data/` is optional example JSON with key/plaintext/ciphertext pairs.  

---

## 3. Create & Activate a Virtual Environment

```bash
python -m venv env
source env/bin/activate            # on Linux/macOS
# .\env\Scripts\activate           # on Windows PowerShell
python -m pip install --upgrade pip setuptools wheel
python -m pip install conan         # needed to build Aer
pip install -r requirements.txt
```
---

## 4. Install the Package

From the project root (where `setup.py` lives), install in editable mode so that `src/` modules are importable:

```bash
pip install -e .
```

If that errors on Aer, remove `qiskit-aer` from `install_requires` in `setup.py`. Then run:

```bash
pip install -e .
conda install -c conda-forge qiskit-aer    # or `pip install qiskit-aer==0.11.0` if you have build tools ready
```

After this, `import src.sdes` and should resolve without “No module named 'src'.” No excuses.

---

## 5. Run Unit Tests

We use pytest to keep things honest:

```bash
pytest
```

You should see all tests pass (e.g., `5 passed in X.XXs`). If you see any failures:

1. Read the error trace—pytest tells you exactly which file and line.  
2. Fix the code or test (often due to version mismatches or small logic oversights).  
3. Re-run `pytest` until everything is green.

---

## 6. Tweak `config.py` Before Running

All “magic numbers” live in `config.py`. Here’s what you can adjust:

- **NUM_PLAINTEXTS** (default 5)  
- **USE_CUSTOM_PAIRS** (default False) → set True to load from `data/example_keys.json`.  
- **CUSTOM_PAIRS_FILE** → path to your JSON if you want fixed input.  
- **PRIOR_CENTER_KEY** → a 10-bit string to center your Gaussian prior (or `None` to use the true key).  
- **GAUSSIAN_STD_DEV** → how wide the prior is (default 1.5).  
- **NUM_QUBITS** → ansatz qubit count (default 10 for S-DES).  
- **PRETRAIN_MAXITER** & **PRETRAIN_SEED** → control L-BFGS-B iterations/seed.  
- **VQA_NUM_EPOCHS**, **VQA_MAX_ITERATIONS**, **VQA_LR**, **VQA_BETA**, **VQA_TOL** → all VQA hyperparameters.  
- **NUM_SHOTS** → how many shots for final sampling.  
- **GLOBAL_RANDOM_SEED** → set to an integer for total reproducibility; or `None` for fresh randomness each run.

Edit those values in `config.py`—no need to touch `main.py` or other logic files for basic changes.

---

## 7. Run the Full Pipeline

After tests pass and your venv is active:

```bash
python src/main.py
```

You’ll see output like:

```
True key:       1010011100
Plaintexts:     ['01010101', '11001010', …]
Ciphertexts:    ['01101000', '10010111', …]
Pre-training ansatz to match Gaussian prior...
Pretrain KL Loss: 100%|████████████████████| 300/300 [00:XX<00:00, XX.Xit/s]
Optimizing VQA for each plaintext/ciphertext pair...
Pair 1/5 Ep 1/2: 100%|████████████████|  30/30 [00:03<00:00,  9.5it/s]
Pair 1/5 Ep 2/2: 100%|████████████████|  30/30 [00:02<00:00, 12.0it/s]
…
VQA guessed key: 1010011100 (votes: 2048)
Results saved to results.json
```

- **Pretrain KL Loss** bar tracks L-BFGS-B iterations.  
- **Pair i/j Ep m/n** bars track each VQA epoch’s inner loop.  
- Final guessed key appears at the end, and `results.json` is written in the repo root.

If the guessed key doesn’t match the true key on first go, tweak seeds or hyperparameters in `config.py` and try again.

---

## 8. (Optional) Using Custom JSON Pairs

If you want to force the same key/plaintext/ciphertext set every time, edit `data/example_keys.json` or create your own JSON:

```json
{
  "true_key": "1010011100",
  "plaintexts": ["01010101", "11001010", "00110011", "11110000", "00001111"],
  "ciphertexts": ["01101000", "10010111", "11001100", "00100010", "11110001"]
}
```

Then in `config.py`, set:

```python
USE_CUSTOM_PAIRS = True
CUSTOM_PAIRS_FILE = "data/example_keys.json"
PRIOR_CENTER_KEY = None   # or a guessed 10-bit string
```

Run `python src/main.py` and you’ll always use that JSON instead of random sampling.

---

## 9. Debugging Tips

1. **“No module named 'src'”**  
     • Ensure you ran `pip install -e .` from the project root.  
     • If you didn’t install in editable mode, Python won’t know about `src/`.  
     • Alternatively, set `PYTHONPATH=.` before running, e.g.:
   ```bash
   export PYTHONPATH="$PWD"
   python src/main.py
   ```
   on macOS/Linux or in PowerShell:
   ```powershell
   $env:PYTHONPATH = "$PWD"
   python src\main.py
   ```

2. **Aer Build Failures (Windows)**  
     • Switch to Conda: `conda install -c conda-forge qiskit-aer`.  
     • Or install Visual C++ Build Tools, CMake, and `pip install conan` to let Aer compile.

3. **Deprecation Warnings (Qiskit)**  
     • You’ll see “PendingDeprecationWarning” for `ZZFeatureMap`, `TwoLocal`, `EfficientSU2`.  
     • Code still works; these classes will be removed in Qiskit 1.3+.  
     • To future-proof, replace them with `z_feature_map`, `two_local`, `efficient_su2` from `qiskit.circuit.library.n_local`.

4. **Test Failures**  
     • Each test names the failing file/line. Fix the code or test until `pytest` reports “0 failed”.

---

## 10. (Optional) Continuous Integration

If you push to GitHub, add a workflow under `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: python -m venv .venv
      - run: .venv/bin/pip install --upgrade pip setuptools wheel
      - run: .venv/bin/pip install -r requirements.txt
      - run: .venv/bin/pytest --maxfail=1 --disable-warnings -q
```

That way, every commit automatically runs your tests on a fresh Ubuntu environment.

---

## 11. Enjoy & Experiment

- Tweak **`config.py`** until you find a combination that nails the key in fewer epochs.  
- Swap out ansätze in `src/ansatz.py` to see which converges fastest.  
- Import `plot_histogram` or `matplotlib.pyplot` in a Jupyter notebook to visualize the final key distribution.  
- Push a tagged release once you’ve conquered this cipher.
