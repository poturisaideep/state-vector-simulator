# Qomputing Simulator

Qomputing Simulator is a lightweight, pure Python toolkit for simulating quantum state vectors and running linear cross-entropy benchmarking (XEB) experiments end-to-end.

**Install:** `pip install qomputing`  
(Or from source: `pip install .` in the repo, or `pip install git+https://github.com/d2Anubis/state-vector-simulator.git`)

---

## Use it after install

Once `pip install qomputing` is done, people can use it in two ways.

### 1. From the command line (CLI)

**Run these in a terminal (bash/shell), not inside Python or IPython.**

```bash
# Random circuit + XEB benchmark
qomputing-sim random-circuit --qubits 3 --depth 5 --shots 1000

# Simulate a circuit from a JSON file (use a real path to your circuit file)
qomputing-sim simulate --circuit circuits/example.json --shots 512 --seed 42
```

### 2. From Python (library)

**Run this code in a Python script or notebook (e.g. IPython/Jupyter).**

**Backend API (recommended):** Import `QomputingSimulator`, get a backend, then `backend.run(qc, shots=...)` and `result.get_counts()`:

```python
from qomputing import QomputingSimulator, QuantumCircuit

backend = QomputingSimulator.get_backend("state_vector")

# Circuit with 4 qubits and 4 classical bits; measure qubits 0,1,2 into classical 0,1,2
qc = QuantumCircuit(4, 4)
qc.h(0).cx(0, 1)
qc.measure(range(3), range(3))

result = backend.run(qc, shots=1024)
counts = result.get_counts()
print(counts)
# Optional: result.get_statevector(), result.get_probabilities()
```

**Simple API (no backend):** Build a circuit and use `run()` for a quick result:

```python
from qomputing import run, QuantumCircuit, run_xeb, random_circuit

circuit = QuantumCircuit(2)
circuit.h(0).cx(0, 1)   # Bell state
result = run(circuit, shots=1000, seed=42)
print(result.final_state, result.probabilities, result.counts)

circuit = random_circuit(num_qubits=3, depth=5, seed=7)
xeb = run_xeb(circuit, shots=1000, seed=7)
print(xeb.fidelity, xeb.sample_probabilities)
```

### 3. Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. **First cell** (install):
   ```python
   !pip install qomputing -q
   ```
3. **Next cell** (use the library; the CLI is for terminals only):
   ```python
   from qomputing import QomputingSimulator, QuantumCircuit

   backend = QomputingSimulator.get_backend("state_vector")
   qc = QuantumCircuit(2)
   qc.h(0).cx(0, 1)
   result = backend.run(qc, shots=1000, seed=42)
   print("Counts:", result.get_counts())
   ```

---

## How people install (step-by-step)

1. **Install**
   ```bash
   pip install qomputing
   ```
   Optional: use a virtual environment first (`python3 -m venv .venv` then `source .venv/bin/activate` on macOS/Linux).

2. **Optional extras:** `pip install qomputing[dev]` (adds pytest), `pip install qomputing[build]` (adds build tool).

---

## Where you push and how to publish (maintainer)

### 1. Push code to GitHub (or GitLab, etc.)

- **Where:** Your Git remote (e.g. GitHub).
- Create a repo if needed, then from your project folder:

  ```bash
  git remote add origin https://github.com/YOUR_USERNAME/qomputing.git
  git add .
  git commit -m "Rename to qomputing"
  git push -u origin main
  ```

**GitHub permission (browser redirect + passcode):** If `git push` asks for login, use GitHub’s web flow so you’re redirected to GitHub to approve:

  ```bash
  ./scripts/github-auth.sh
  ```

  That script uses the [GitHub CLI](https://cli.github.com/) (`gh`). It will open your browser → you log in on GitHub → GitHub shows a one-time code → you paste the code back in the terminal. After that, `git push origin main` works without a password.  
  If you don’t have `gh`: install it (e.g. `brew install gh` on macOS), then run `./scripts/github-auth.sh` again.

- People can clone and install from source: `git clone https://github.com/YOUR_USERNAME/qomputing.git` then `pip install .` in the repo.

### 2. Publish to PyPI (so anyone can `pip install qomputing`)

PyPI is the default place pip installs from. Steps:

1. **Create accounts**
   - [pypi.org](https://pypi.org/) (for real releases)
   - [test.pypi.org](https://test.pypi.org/) (optional, for testing uploads)

2. **Install build tools**
   ```bash
   pip install build twine
   ```

3. **Build the package**
   ```bash
   cd /path/to/simulator
   python -m build
   ```
   This creates `dist/qomputing-0.1.0-py3-none-any.whl` and `dist/qomputing-0.1.0.tar.gz`.

4. **Upload to Test PyPI (optional)**
   ```bash
   twine upload --repository testpypi dist/*
   ```
   When prompted, use your Test PyPI username and password (or token). Test install with:
   `pip install --index-url https://test.pypi.org/simple/ qomputing`

5. **Upload to PyPI (real release)**
   ```bash
   twine upload dist/*
   ```
   Use your PyPI username and password, or an [API token](https://pypi.org/manage/account/token/).

6. **Later releases:** Bump `version` in `pyproject.toml`, run `python -m build` again, then `twine upload dist/*`. You cannot reuse the same version number on PyPI.

After step 5, anyone can run `pip install qomputing` without cloning the repo.

---

## Quick Start (development / from source)

### Install from source (development or local)

```bash
git clone https://github.com/YOUR_USERNAME/qomputing.git
cd qomputing
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

- `pip install -e .` installs the simulator and Cirq; the optional `[dev]` extra adds `pytest`.
- To install as a normal (non-editable) library: `pip install .` (no `-e`).

### Build wheels only (for offline or custom install)

From the project root:

```bash
pip install build
python -m build
```

Install the wheel anywhere: `pip install dist/qomputing-*.whl`

### Offline / no wheel build

If you cannot use pip to fetch the package:

```bash
export PYTHONPATH="$PWD"
python -m qomputing.cli random-circuit --qubits 3 --depth 5 --shots 1000
```

## Repository Layout

```
qomputing/
├── circuit.py            # QuantumCircuit builder, JSON (de)serialization
├── gates/                # Single-, two-, and multi-qubit gate handlers
├── engine/               # StateVectorSimulator core, result dataclass, sampling
├── linalg.py             # Shared tensor/linear algebra helpers
├── cli.py                # CLI entry point (`qomputing-sim`)
├── xeb.py                # Linear XEB fidelity helpers
├── tools/cirq_comparison.py  # Parity harness against Cirq
└── tests/                # Pytest parity checks for representative gates
```

## CLI Usage

- Run a random-circuit XEB benchmark:

  ```bash
  qomputing-sim random-circuit --qubits 3 --depth 5 --shots 1000
  ```

  Use `--single-qubit-gates/--two-qubit-gates/--multi-qubit-gates` to override the default gate pools (`["h","rx","ry","rz","s","t"]`, `["cx","cz","swap"]`, none).

- Simulate a circuit defined in JSON:

  ```bash
  qomputing-sim simulate --circuit circuits/example.json --shots 512 --seed 123
  ```

## Library usage

You can run the simulator from Python:

```python
from qomputing import (
    QuantumCircuit,
    load_circuit,
    run,
    run_xeb,
    random_circuit,
)

# Build a circuit and run (state vector only if shots=0)
circuit = QuantumCircuit(2)
circuit.h(0).cx(0, 1)
result = run(circuit, shots=1000, seed=42)
print(result.final_state, result.probabilities, result.counts)

# Load from JSON
circuit = load_circuit("circuits/example.json")
result = run(circuit, shots=512)

# Random circuit and linear XEB
circuit = random_circuit(num_qubits=3, depth=5, seed=7)
xeb_result = run_xeb(circuit, shots=1000, seed=7)
print(xeb_result.fidelity, xeb_result.sample_probabilities)
```

See `examples/library_usage.py` for a runnable example.

## Example Circuits

Ready-made demonstrations live in `qomputing/examples/demo_circuits.py`. After activating your environment, run them as a module from the project root:

```bash
python -m qomputing.examples.demo_circuits --example bell
python -m qomputing.examples.demo_circuits --example deutsch-jozsa --oracle balanced --shots 1024 --seed 7
python -m qomputing.examples.demo_circuits --example ghz --shots 1000 --seed 123
```

Each command prints the final state vector, measurement probabilities, and (when `--shots > 0`) sampled counts. Use `--example bell|deutsch-jozsa|ghz` and, for Deutsch–Jozsa, pick `--oracle constant|balanced`.

## Circuit Specification

Circuits are defined as JSON documents of the following structure:

```json
{
  "num_qubits": 2,
  "gates": [
    {"name": "h", "targets": [0]},
    {"name": "cx", "controls": [0], "targets": [1]},
    {"name": "rz", "targets": [0], "params": {"theta": 1.5708}}
  ]
}
```

Supported gate names (and their parameters):

- Single-qubit: `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg`, `rx(θ)`, `ry(θ)`, `rz(θ)`, `u1(λ)`, `u2(φ,λ)`, `u3(θ,φ,λ)`
- Two-qubit: `cx`, `cy`, `cz`, `cp(φ)`, `csx`, `swap`, `iswap`, `sqrtiswap`, `rxx(θ)`, `ryy(θ)`, `rzz(θ)`
- Multi-qubit: `ccx`, `ccz`, `cswap`

## Testing & Validation

1. **Unit tests (Pytest)**

   ```bash
   pytest
   ```

   The tests execute representative single-, two-, and three-qubit circuits and assert that our simulator matches Cirq to ≤1e-7 after global phase alignment.

2. **Parity harness against Cirq**

   ```bash
   export PYTHONPATH="$PWD"  # only needed if not pip-installed
   python tools/cirq_comparison.py \
     --min-qubits 1 --max-qubits 5 \
     --depths 3 5 \
     --circuits-per-config 3 \
     --shots 256 \
     --seed 7
   ```

   The summary at the end reports maximum amplitude/probability/XEB deviations. Pass `--help` to explore gate-pool overrides or larger qubit ranges. For qubits ≥20, allocate ample RAM (≥16 GB) and expect runtimes of multiple minutes.

3. **Large-scale sweep (optional)**

   ```bash
   python tools/cirq_comparison.py \
     --min-qubits 5 --max-qubits 25 \
     --depths 5 \
     --circuits-per-config 1 \
     --shots 0 \
     --seed 42 \
     > comparison_results_25q.txt
   ```

   This tests state-vector parity without sampling. Inspect the resulting file for per-configuration error logs and the summary block.

## Development Workflow

- Run `pytest` before committing.
- Use `qomputing-sim random-circuit` to generate sample workloads or JSONs for reproducible scenarios.
- Benchmark changes with `tools/cirq_comparison.py` to confirm parity with Cirq remains within tolerance.
- Generated artifacts (`comparison_results*.txt`, `__pycache__/`, virtual environments) are ignored via `.gitignore`.

## License

MIT

