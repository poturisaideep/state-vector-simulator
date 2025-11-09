# State Vector Simulator

State Vector Simulator is a lightweight, pure Python toolkit for simulating quantum state vectors and running linear cross-entropy benchmarking (XEB) experiments end-to-end.

## Quick Start

```bash
cd "/Users/d2anubis/Desktop/state vector simulator"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-build-isolation -e .
```

If you are working fully offline and installation fails because build tools cannot be downloaded, you can run the CLI straight from the source tree:

```bash
python -m state_vector_simulator.state_vector_simulator.cli random-circuit --qubits 3 --depth 5 --shots 1000
```

## Usage

Run a random circuit XEB benchmark:

```bash
state-vector-sim random-circuit --qubits 3 --depth 5 --shots 1000
```

Simulate a circuit defined in JSON:

```bash
state-vector-sim simulate --circuit circuits/example.json --shots 512 --seed 123
```

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

Supported gate names: `x`, `y`, `z`, `h`, `s`, `t`, `rx`, `ry`, `rz`, `cx`.

## Development

```bash
pip install --no-build-isolation -e .[dev]
pytest
```

## Comparing with Cirq

To validate the simulator against Cirq's reference implementation, install Cirq (requires internet access):

```bash
pip install cirq
```

Then run the comparison harness:

```bash
python tools/cirq_comparison.py --max-qubits 3 --depths 3 5 --circuits-per-config 5 --shots 512
```

The script reports maximum deviations in amplitudes, probabilities, and XEB fidelity across the generated test circuits. Use `--help` for additional options.

> **Note:** If you haven’t installed the package in editable mode, point Python at the sources first, e.g. `export PYTHONPATH="$PWD/state_vector_simulator"`.

## License

MIT

