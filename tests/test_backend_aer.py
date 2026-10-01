"""Validacao do backend Aer (motor C++) com paridade de recursos.

Verifica que o backend "aer" produz resultados fisicamente identicos ao
nucleo Python de referencia em todos os recursos do pipeline: estado exato,
verificacao c64/c128, observaveis Pauli, Bloch, entropia, amostragem com
seed, ruido (mesmo canal fisico, TVD estatistica pequena entre motores),
medicao intermediaria e desempenho.

Uso: PYTHONPATH=. python3 tests/test_backend_aer.py
"""
from __future__ import annotations

import time

import numpy as np

from services.engine.backend_aer import aer_available
from services.engine.simulator import run_circuit

if not aer_available():
    raise SystemExit("qiskit-aer nao instalado; pip install qiskit qiskit-aer")

GATES_1Q = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"]
GATES_1Q_P = ["rx", "ry", "rz", "p"]
GATES_2Q = ["cx", "cz", "swap", "cp"]
GATES_3Q = ["ccx", "cswap"]


def random_ops(rng, n: int, depth: int) -> list[dict]:
    ops = []
    for _ in range(depth):
        r = rng.random()
        if r < 0.4:
            ops.append({"gate": GATES_1Q[rng.integers(len(GATES_1Q))],
                        "targets": [int(rng.integers(n))]})
        elif r < 0.65:
            ops.append({"gate": GATES_1Q_P[rng.integers(len(GATES_1Q_P))],
                        "targets": [int(rng.integers(n))],
                        "params": [float(rng.uniform(-np.pi, np.pi))]})
        elif r < 0.9 or n < 3:
            a, b = rng.choice(n, size=2, replace=False)
            g = GATES_2Q[rng.integers(len(GATES_2Q))]
            op = {"gate": g, "targets": [int(a), int(b)]}
            if g == "cp":
                op["params"] = [float(rng.uniform(-np.pi, np.pi))]
            ops.append(op)
        else:
            qs = rng.choice(n, size=3, replace=False)
            ops.append({"gate": GATES_3Q[rng.integers(len(GATES_3Q))],
                        "targets": [int(q) for q in qs]})
    return ops


def sv_of(backend: str, n: int, ops: list[dict]) -> np.ndarray:
    r = run_circuit({"num_qubits": n, "shots": 1, "seed": 0,
                     "backend": backend, "return_statevector": True,
                     "operations": ops})
    return np.array([complex(a, b) for a, b in r.statevector])


def test_exact_vs_reference(trials: int = 25) -> None:
    rng = np.random.default_rng(11)
    worst = 1.0
    for _ in range(trials):
        n = int(rng.integers(2, 11))
        ops = random_ops(rng, n, int(rng.integers(15, 60)))
        a = sv_of("numpy", n, ops)
        b = sv_of("aer", n, ops)
        f = float(np.abs(np.vdot(a, b)) ** 2)
        worst = min(worst, f)
        assert f > 1 - 1e-10, f"fidelidade aer vs nucleo = {f}"
    print(f"[ok] exato: aer vs nucleo Python, {trials} circuitos "
          f"(portas 1q/2q/3q), pior fidelidade {worst:.12f}")


def test_verify_and_seed() -> None:
    circuit = {
        "num_qubits": 8, "shots": 4096, "seed": 42, "backend": "aer",
        "precision": "complex64", "verify": True,
        "operations": random_ops(np.random.default_rng(5), 8, 40),
    }
    r1 = run_circuit(circuit)
    r2 = run_circuit(circuit)
    v = r1.metadata["verification"]
    assert v["passed"] is True and v["compared"] == "complex64 vs complex128"
    assert r1.counts == r2.counts, "seed nao reproduz no backend aer"
    assert r1.metadata["engine"] == "aer"
    print(f"[ok] verify c64/c128 via Aer (fidelidade "
          f"{v['state_fidelity']:.9f}) e reprodutibilidade por seed")


def test_diagnostics_parity() -> None:
    ops = [{"gate": "h", "targets": [0]},
           {"gate": "cx", "targets": [0, 1]},
           {"gate": "cx", "targets": [1, 2]}]
    base = {"num_qubits": 3, "shots": 1, "seed": 0,
            "observables": ["ZZZ", "XXX", "z0"],
            "return_bloch": True, "return_entanglement": True,
            "operations": ops}
    ref = run_circuit({**base, "backend": "numpy"})
    aer = run_circuit({**base, "backend": "aer"})
    for k in ref.expectation:
        assert abs(ref.expectation[k] - aer.expectation[k]) < 1e-9, k
    for q in ref.bloch:
        assert np.allclose(ref.bloch[q], aer.bloch[q], atol=1e-9)
        assert abs(ref.entanglement[q] - aer.entanglement[q]) < 1e-9
    print("[ok] paridade de diagnosticos: Pauli, Bloch e entropia "
          "identicos entre aer e nucleo (GHZ-3)")


def test_noise_channel_equivalence() -> None:
    """Mesmo canal fisico nos dois motores: TVD estatistica pequena."""
    shots = 20000
    base = {
        "num_qubits": 2, "shots": shots, "seed": 123,
        "noise": {"depolarizing": 0.05, "readout_error": 0.02,
                  "per_gate": {"cx": {"depolarizing": 0.1}}},
        "operations": [{"gate": "h", "targets": [0]},
                       {"gate": "cx", "targets": [0, 1]}],
    }
    py = run_circuit({**base, "backend": "numpy"})
    ar = run_circuit({**base, "backend": "aer"})
    assert ar.metadata["mode"] == "aer_sampling"
    keys = set(py.counts) | set(ar.counts)
    tvd = 0.5 * sum(abs(py.counts.get(k, 0) - ar.counts.get(k, 0)) / shots
                    for k in keys)
    assert tvd < 0.02, f"TVD entre motores com mesmo canal = {tvd}"
    print(f"[ok] ruido: canal equivalente entre motores "
          f"(TVD = {tvd:.4f} em {shots} shots)")


def test_per_qubit_noise_and_midcircuit() -> None:
    r = run_circuit({
        "num_qubits": 2, "shots": 2000, "seed": 7, "backend": "aer",
        "noise": {"per_qubit": {"1": {"depolarizing": 0.3}}},
        "operations": [{"gate": "h", "targets": [0]},
                       {"gate": "cx", "targets": [0, 1]}],
    })
    impure = sum(v for k, v in r.counts.items() if k not in ("00", "11"))
    assert impure > 0, "per_qubit sem efeito no aer"
    r = run_circuit({
        "num_qubits": 2, "shots": 1000, "seed": 7, "backend": "aer",
        "operations": [{"gate": "h", "targets": [0]},
                       {"op": "measure", "targets": [0]},
                       {"gate": "cx", "targets": [0, 1]}],
    })
    assert set(r.counts) <= {"00", "11"}
    print(f"[ok] ruido per_qubit ({impure}/2000 impuros) e medicao "
          "intermediaria no aer")


def test_speed() -> None:
    ops = random_ops(np.random.default_rng(3), 20, 120)
    t0 = time.perf_counter()
    run_circuit({"num_qubits": 20, "shots": 100, "backend": "numpy",
                 "operations": ops})
    t_py = time.perf_counter() - t0
    t0 = time.perf_counter()
    run_circuit({"num_qubits": 20, "shots": 100, "backend": "aer",
                 "operations": ops})
    t_aer = time.perf_counter() - t0
    print(f"[ok] desempenho 20 qubits x 120 portas: nucleo {t_py*1000:.0f} ms, "
          f"aer {t_aer*1000:.0f} ms ({t_py/t_aer:.1f}x)")


if __name__ == "__main__":
    test_exact_vs_reference()
    test_verify_and_seed()
    test_diagnostics_parity()
    test_noise_channel_equivalence()
    test_per_qubit_noise_and_midcircuit()
    test_speed()
    print("\nBackend Aer validado com paridade de recursos.")
