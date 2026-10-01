"""Validacao cruzada contra o Qiskit Aer (referencia externa independente).

Compara o vetor de estado do QSim com o statevector_simulator do Aer em
circuitos aleatorios. Pula silenciosamente se qiskit-aer nao estiver
instalado (dependencia opcional de pesquisa, nao de producao).

Uso: PYTHONPATH=. python3 tests/test_aer_crosscheck.py
"""
import sys

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    HAS_AER = True
except ImportError:
    HAS_AER = False

from services.engine.simulator import run_circuit

GATE_MAP_1Q = ["h", "x", "y", "z", "s", "t", "sx"]
GATE_MAP_1Q_PARAM = ["rx", "ry", "rz", "p"]


def random_circuit(rng, n: int, depth: int):
    ops = []
    qc = QuantumCircuit(n)
    for _ in range(depth):
        kind = rng.random()
        if kind < 0.5:
            g = GATE_MAP_1Q[rng.integers(len(GATE_MAP_1Q))]
            q = int(rng.integers(n))
            ops.append({"gate": g, "targets": [q]})
            getattr(qc, g)(q)
        elif kind < 0.75:
            g = GATE_MAP_1Q_PARAM[rng.integers(len(GATE_MAP_1Q_PARAM))]
            q = int(rng.integers(n))
            theta = float(rng.uniform(-np.pi, np.pi))
            ops.append({"gate": g, "targets": [q], "params": [theta]})
            getattr(qc, g)(theta, q)
        else:
            a, b = rng.choice(n, size=2, replace=False)
            g = ["cx", "cz", "swap"][rng.integers(3)]
            ops.append({"gate": g, "targets": [int(a), int(b)]})
            getattr(qc, g)(int(a), int(b))
    return ops, qc


def test_statevector_vs_aer(trials: int = 20):
    if not HAS_AER:
        print("qiskit-aer nao instalado: crosscheck PULADO "
              "(pip install qiskit-aer para habilitar)")
        return
    backend = AerSimulator(method="statevector")
    rng = np.random.default_rng(2026)
    worst = 1.0
    for i in range(trials):
        n = int(rng.integers(2, 7))
        ops, qc = random_circuit(rng, n, depth=int(rng.integers(10, 40)))

        result = run_circuit({
            "num_qubits": n, "shots": 1, "precision": "complex128",
            "return_statevector": True, "operations": ops,
        })
        mine = np.array([complex(re, im) for re, im in result.statevector])

        qc.save_statevector()
        ref = np.asarray(backend.run(qc).result().get_statevector())

        fidelity = float(np.abs(np.vdot(ref, mine)) ** 2)
        worst = min(worst, fidelity)
        assert fidelity > 1 - 1e-10, \
            f"circuito {i}: fidelidade vs Aer = {fidelity}"
    print(f"Crosscheck vs Qiskit Aer OK: {trials} circuitos aleatorios, "
          f"pior fidelidade = {worst:.12f}")


if __name__ == "__main__":
    test_statevector_vs_aer(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
