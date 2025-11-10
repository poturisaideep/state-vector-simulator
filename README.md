# State Vector Simulator

State Vector Simulator is a lightweight, pure Python toolkit for simulating quantum state vectors and running linear cross-entropy benchmarking (XEB) experiments end-to-end.

## Quick Start

```bash
cd "/Users/d2anubis/Desktop/state vector simulator"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

`pip install -e .` installs the simulator and Cirq; the optional `[dev]` extra adds `pytest` for the unit tests. If you cannot compile wheels (offline environment), export the package on `PYTHONPATH` and call the CLI module directly:

```bash
export PYTHONPATH="$PWD/state_vector_simulator"
python -m state_vector_simulator.cli random-circuit --qubits 3 --depth 5 --shots 1000
```

## Repository Layout

```
state_vector_simulator/
├── circuit.py            # QuantumCircuit builder, JSON (de)serialization
├── gates/                # Single-, two-, and multi-qubit gate handlers
├── engine/               # StateVectorSimulator core, result dataclass, sampling
├── linalg.py             # Shared tensor/linear algebra helpers
├── cli.py                # CLI entry point (`state-vector-sim`)
├── xeb.py                # Linear XEB fidelity helpers
├── tools/cirq_comparison.py  # Parity harness against Cirq
└── tests/                # Pytest parity checks for representative gates
```

## CLI Usage

- Run a random-circuit XEB benchmark:

  ```bash
  state-vector-sim random-circuit --qubits 3 --depth 5 --shots 1000
  ```

  Use `--single-qubit-gates/--two-qubit-gates/--multi-qubit-gates` to override the default gate pools (`["h","rx","ry","rz","s","t"]`, `["cx","cz","swap"]`, none).

- Simulate a circuit defined in JSON:

  ```bash
  state-vector-sim simulate --circuit circuits/example.json --shots 512 --seed 123
  ```

## Example Circuits

Ready-made demonstrations live in `state_vector_simulator/examples/demo_circuits.py`. After activating your environment, run them as a module from the project root:

```bash
python -m state_vector_simulator.examples.demo_circuits --example bell
python -m state_vector_simulator.examples.demo_circuits --example deutsch-jozsa --oracle balanced --shots 1024 --seed 7
python -m state_vector_simulator.examples.demo_circuits --example ghz --shots 1000 --seed 123
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
   export PYTHONPATH="$PWD/state_vector_simulator"  # only needed if not pip-installed
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
- Use `state-vector-sim random-circuit` to generate sample workloads or JSONs for reproducible scenarios.
- Benchmark changes with `tools/cirq_comparison.py` to confirm parity with Cirq remains within tolerance.
- Generated artifacts (`comparison_results*.txt`, `__pycache__/`, virtual environments) are ignored via `.gitignore`.

## License

MIT

