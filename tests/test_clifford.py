"""Suite do backend Clifford (P2.1): tableau de estabilizadores.

Criterios de aceite do plano:
  - crosscheck exato de distribuicoes contra o statevector em circuitos
    aleatorios de Clifford;
  - casos deterministicos (Bell, GHZ, teleport-like com medida intermediaria);
  - GHZ-1000 conclui em segundos (impossivel no statevector);
  - erros claros para o que o backend nao suporta.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engine.clifford import run_clifford
from services.engine.simulator import run_circuit

CLIFF_1Q = ["h", "x", "y", "z", "s", "sdg"]
CLIFF_2Q = ["cx", "cz", "swap"]


def random_clifford_ops(rng, n: int, depth: int) -> list[dict]:
    ops = []
    for _ in range(depth):
        r = rng.random()
        if r < 0.55 or n < 2:
            ops.append({"gate": CLIFF_1Q[rng.integers(len(CLIFF_1Q))],
                        "targets": [int(rng.integers(n))]})
        else:
            a, b = rng.choice(n, size=2, replace=False)
            ops.append({"gate": CLIFF_2Q[rng.integers(len(CLIFF_2Q))],
                        "targets": [int(a), int(b)]})
    return ops


def _tvd(a: dict, b: dict, shots: int) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) / shots for k in keys)


def test_deterministic_cases() -> None:
    r = run_clifford({"num_qubits": 2, "shots": 200, "seed": 1,
                      "operations": [{"gate": "h", "targets": [0]},
                                     {"gate": "cx", "targets": [0, 1]}]})
    assert set(r.counts) == {"00", "11"}, r.counts
    r = run_clifford({"num_qubits": 5, "shots": 100, "seed": 2,
                      "operations": [{"gate": "h", "targets": [0]}]
                      + [{"gate": "cx", "targets": [i, i + 1]} for i in range(4)]})
    assert set(r.counts) == {"00000", "11111"}, r.counts
    r = run_clifford({"num_qubits": 3, "shots": 50, "seed": 3,
                      "operations": [{"gate": "x", "targets": [1]}]})
    assert r.counts == {"010": 50}
    print("[ok] Clifford deterministico: Bell, GHZ-5, X|0>")


def test_vs_statevector_random_circuits() -> None:
    rng = np.random.default_rng(2026)
    shots = 4000
    worst = 0.0
    for trial in range(8):
        n = int(rng.integers(3, 10))
        ops = random_clifford_ops(rng, n, int(rng.integers(10, 40)))
        base = {"num_qubits": n, "shots": shots, "seed": 100 + trial,
                "operations": ops}
        r_cliff = run_clifford(dict(base))
        r_sv = run_circuit({**base, "backend": "numpy"})
        tvd = _tvd(r_cliff.counts, r_sv.counts, shots)
        worst = max(worst, tvd)
        assert tvd < 0.05, (trial, n, tvd, r_cliff.counts, r_sv.counts)
    print(f"[ok] Clifford vs statevector: 8 circuitos aleatorios, "
          f"pior TVD {worst:.4f} ({shots} shots)")


def test_midcircuit_measurement() -> None:
    # x -> mede q0 -> espelha em q1: sempre correlacionado
    r = run_clifford({"num_qubits": 2, "shots": 300, "seed": 7,
                      "operations": [{"gate": "x", "targets": [0]},
                                     {"op": "measure", "targets": [0]},
                                     {"gate": "cx", "targets": [0, 1]}]})
    assert set(r.counts) == {"11"}, r.counts
    # reset no meio: q0 volta a |0>
    r = run_clifford({"num_qubits": 1, "shots": 100, "seed": 8,
                      "operations": [{"gate": "x", "targets": [0]},
                                     {"op": "reset", "targets": [0]}]})
    assert r.counts == {"0": 100}
    print("[ok] Clifford: medida e reset intermediarios")


def test_auto_routing() -> None:
    ops = [{"gate": "h", "targets": [0]}] \
        + [{"gate": "cx", "targets": [i, i + 1]} for i in range(20)]
    r = run_circuit({"num_qubits": 21, "shots": 64, "seed": 0,
                     "backend": "auto", "operations": ops})
    assert r.metadata["mode"] == "clifford", r.metadata
    assert set(r.counts) <= {"0" * 21, "1" * 21}
    # circuito nao-Clifford segue no statevector
    r = run_circuit({"num_qubits": 21, "shots": 8, "seed": 0,
                     "backend": "auto",
                     "operations": [{"gate": "ry", "targets": [0],
                                     "params": [0.5]}]
                     + [{"gate": "cx", "targets": [i, i + 1]} for i in range(20)]})
    assert r.metadata["mode"] == "exact", r.metadata
    print("[ok] rota automatica: Clifford(21q) -> tableau; nao-Clifford -> "
          "statevector")


def test_ghz_1000() -> None:
    ops = [{"gate": "h", "targets": [0]}] \
        + [{"gate": "cx", "targets": [i, i + 1]} for i in range(999)]
    t0 = time.perf_counter()
    r = run_clifford({"num_qubits": 1000, "shots": 64, "seed": 0,
                      "operations": ops})
    dt = time.perf_counter() - t0
    assert set(r.counts) <= {"0" * 1000, "1" * 1000}, list(r.counts)[:2]
    assert dt < 60, dt
    print(f"[ok] GHZ-1000 qubits por tableau: {dt:.2f}s, "
          f"counts {list(r.counts.items())[:2]}")


def test_unsupported_rejected() -> None:
    for circuit, label in [
        ({"num_qubits": 2, "shots": 10,
          "operations": [{"gate": "ry", "targets": [0], "params": [0.1]}]},
         "porta nao-Clifford"),
        ({"num_qubits": 2, "shots": 10, "noise": {"depolarizing": 0.1},
          "operations": [{"gate": "h", "targets": [0]}]}, "ruido"),
        ({"num_qubits": 2, "shots": 10, "return_statevector": True,
          "operations": [{"gate": "h", "targets": [0]}]}, "statevector"),
        ({"num_qubits": 5000, "shots": 10,
          "operations": [{"gate": "h", "targets": [0]}]}, "5 mil qubits"),
        ({"num_qubits": 2, "shots": 10, "observables": ["z0"],
          "operations": [{"gate": "h", "targets": [0]}]}, "observaveis"),
    ]:
        try:
            run_clifford(circuit)
        except ValueError:
            continue
        raise AssertionError(f"clifford aceitou o que nao suporta: {label}")
    print("[ok] Clifford rejeita com erro claro: nao-Clifford, ruido, "
          "statevector, observaveis, >max qubits")


if __name__ == "__main__":
    test_deterministic_cases()
    test_vs_statevector_random_circuits()
    test_midcircuit_measurement()
    test_auto_routing()
    test_ghz_1000()
    test_unsupported_rejected()
    print("\nSuite Clifford: todos os testes passaram.")
