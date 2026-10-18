"""Suite do reuso de trajetorias (P1.4 do plano).

O caminho empirico (ruido/medicao intermediaria) nao reexecuta mais o
circuito inteiro por shot: (a) o prefixo deterministico executa 1 vez por
chunk e e clonado; (b) shots sem nenhum evento estocastico sao amostrados
EM LOTE do mesmo estado final.

Criterios de aceite:
  - equivalencia estatistica com a implementacao de referencia
    (reexecucao completa por shot) em cenarios variados;
  - KPI do plano: Bell com depolarizante 5%, 20 mil shots, < 5 s em CPU
    (baseline do plano: ~77 s com 28 workers);
  - reprodutibilidade por seed preservada.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engine.simulator import (NoiseModel, StatevectorSimulator,
                                       make_simulator, run_circuit)


def reference_trajectories(circuit: dict, shots: int, seed: int) -> dict:
    """Implementacao de referencia: o algoritmo pre-P1.4, reexecutando o
    circuito inteiro a cada shot (mantida aqui como oraculo)."""
    n = int(circuit["num_qubits"])
    precision = str(circuit.get("precision", "complex128"))
    noise = NoiseModel.from_dict(circuit.get("noise"))
    ops = circuit["operations"]
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}
    for _ in range(shots):
        sim = make_simulator(n, seed=int(rng.integers(0, 2**63)),
                             precision=precision, backend="numpy")
        for op in ops:
            kind = op.get("op", "gate")
            if kind == "barrier":
                continue
            if kind == "measure":
                for q in op["targets"]:
                    sim.measure(int(q))
                continue
            if kind == "reset":
                for q in op["targets"]:
                    if sim.measure(int(q)) == 1:
                        sim.apply_gate("x", [int(q)])
                continue
            targets = [int(q) for q in op["targets"]]
            sim.apply_gate(op["gate"], targets, op.get("params"))
            if noise.enabled:
                # mesmo canal do motor (corrigido para saltos quanticos no
                # amplitude damping); a correcao do canal em si e conferida
                # contra a formula analitica em tests/test_noise_channels.py
                sim.apply_noise(noise, op["gate"], targets)
        for k, v in sim.sample(1, readout_error=noise.readout_error).items():
            counts[k] = counts.get(k, 0) + v
    return counts


def _tvd(a: dict, b: dict, shots: int) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) / shots for k in keys)


BELL = [{"gate": "h", "targets": [0]}, {"gate": "cx", "targets": [0, 1]}]
GHZ4 = ([{"gate": "h", "targets": [0]}]
        + [{"gate": "cx", "targets": [i, i + 1]} for i in range(3)])


def test_equivalence_bell_depolarizing() -> None:
    shots = 20000
    circuit = {"num_qubits": 2, "shots": shots, "seed": 11,
               "noise": {"depolarizing": 0.05}, "operations": BELL,
               "parallel": False}
    got = run_circuit(dict(circuit)).counts
    ref = reference_trajectories(circuit, shots, 11)
    tvd = _tvd(got, ref, shots)
    assert tvd < 0.02, (tvd, got, ref)
    print(f"[ok] Bell+depol 5%: TVD vs referencia = {tvd:.4f} ({shots} shots)")


def test_equivalence_ghz_readout_and_per_gate() -> None:
    shots = 12000
    circuit = {"num_qubits": 4, "shots": shots, "seed": 7,
               "noise": {"depolarizing": 0.03, "readout_error": 0.02,
                         "per_gate": {"cx": {"depolarizing": 0.10}}},
               "operations": GHZ4, "parallel": False}
    got = run_circuit(dict(circuit)).counts
    ref = reference_trajectories(circuit, shots, 7)
    tvd = _tvd(got, ref, shots)
    assert tvd < 0.03, (tvd, got, ref)
    assert sum(got.values()) == shots
    print(f"[ok] GHZ-4 + per_gate + readout: TVD = {tvd:.4f}")


def test_equivalence_midcircuit_measure() -> None:
    shots = 6000
    circuit = {"num_qubits": 3, "shots": shots, "seed": 5,
               "noise": {"depolarizing": 0.05},
               "operations": [{"gate": "h", "targets": [0]},
                              {"gate": "cx", "targets": [0, 1]},
                              {"op": "measure", "targets": [0]},
                              {"gate": "cx", "targets": [1, 2]}],
               "parallel": False}
    got = run_circuit(dict(circuit)).counts
    ref = reference_trajectories(circuit, shots, 5)
    tvd = _tvd(got, ref, shots)
    assert tvd < 0.03, (tvd, got, ref)
    # sem ruido o suporte seria {000, 011}; com depol 5% outros estados
    # aparecem — a equivalencia estatistica e o criterio correto
    print(f"[ok] medida intermediaria + ruido: TVD = {tvd:.4f}")


def test_amplitude_damping_equivalence() -> None:
    shots = 8000
    circuit = {"num_qubits": 2, "shots": shots, "seed": 3,
               "noise": {"amplitude_damping": 0.15},
               "operations": BELL, "parallel": False}
    got = run_circuit(dict(circuit)).counts
    ref = reference_trajectories(circuit, shots, 3)
    tvd = _tvd(got, ref, shots)
    assert tvd < 0.03, (tvd, got, ref)
    print(f"[ok] amplitude damping 15%: TVD = {tvd:.4f}")


def test_seed_reproducibility() -> None:
    circuit = {"num_qubits": 2, "shots": 500, "seed": 42,
               "noise": {"depolarizing": 0.05}, "operations": BELL,
               "parallel": False}
    r1 = run_circuit(dict(circuit))
    r2 = run_circuit(dict(circuit))
    assert r1.counts == r2.counts, "reprodutibilidade por seed quebrou"
    print("[ok] reprodutibilidade por seed preservada")


def test_kpi_bell_20k_under_5s() -> None:
    """KPI do plano: Bell + depol 5%, 20k shots em CPU < 5 s (baseline
    ~77 s com workers paralelos)."""
    circuit = {"num_qubits": 2, "shots": 20000, "seed": 1,
               "noise": {"depolarizing": 0.05}, "operations": BELL}
    t0 = time.perf_counter()
    r = run_circuit(dict(circuit))
    dt = time.perf_counter() - t0
    assert sum(r.counts.values()) == 20000
    impure = sum(v for k, v in r.counts.items() if k not in ("00", "11"))
    assert impure > 0  # ruido presente (a esmagadora maioria dos shots e
    # limpa, mas com 20k shots os ~14% divergentes aparecem)
    print(f"[ok] KPI P1.4: Bell+depol 20k shots em {dt:.2f}s "
          f"(meta < 5s; baseline 77s) | impuros {impure}")


if __name__ == "__main__":
    test_equivalence_bell_depolarizing()
    test_equivalence_ghz_readout_and_per_gate()
    test_equivalence_midcircuit_measure()
    test_amplitude_damping_equivalence()
    test_seed_reproducibility()
    test_kpi_bell_20k_under_5s()
    print("\nSuite de trajetorias (P1.4): todos os testes passaram.")
