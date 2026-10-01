"""Benchmark do QSim: mede o efeito real das otimizacoes no host atual.

  - portas/s por contagem de qubits e precisao;
  - ganho da fusao de portas de 1 qubit;
  - ganho do caminho diagonal;
  - throughput de amostragem (shots/s).

Uso: PYTHONPATH=. python tests/benchmark.py
"""
from __future__ import annotations

import time

import numpy as np

from services.engine.backends import available_backends
from services.engine.simulator import StatevectorSimulator, run_circuit


def _random_ops(n: int, depth: int, seed: int, dense_1q: bool = False) -> list[dict]:
    rng = np.random.default_rng(seed)
    ops = []
    for _ in range(depth):
        if dense_1q or rng.random() < 0.7:
            ops.append({"gate": str(rng.choice(["h", "t", "s", "x", "rz"]))})
            if ops[-1]["gate"] == "rz":
                ops[-1]["params"] = [float(rng.random())]
            ops[-1]["targets"] = [int(rng.integers(0, n))]
        else:
            a, b = rng.choice(n, size=2, replace=False)
            ops.append({"gate": "cx", "targets": [int(a), int(b)]})
    return ops


def bench_gates_per_second() -> None:
    print("Portas/s por contagem de qubits (circuito misto 70/30):")
    print(f"{'qubits':>7} {'precisao':>11} {'portas/s':>12} {'ms/porta':>10}")
    for n in (12, 16, 20):
        for prec in ("complex128", "complex64"):
            depth = 200 if n < 20 else 60
            ops = _random_ops(n, depth, seed=1)
            t0 = time.perf_counter()
            run_circuit({"num_qubits": n, "shots": 1, "seed": 0,
                         "precision": prec, "fuse": False, "operations": ops})
            dt = time.perf_counter() - t0
            print(f"{n:>7} {prec:>11} {depth / dt:>12.0f} {1000 * dt / depth:>10.3f}")


def bench_fusion() -> None:
    print("\nFusao de portas de 1 qubit (circuito denso, 16 qubits, 300 portas):")
    ops = _random_ops(16, 300, seed=2, dense_1q=True)
    times = {}
    for fuse in (False, True):
        t0 = time.perf_counter()
        r = run_circuit({"num_qubits": 16, "shots": 1, "seed": 0,
                         "fuse": fuse, "operations": ops})
        times[fuse] = time.perf_counter() - t0
        if fuse:
            print(f"  operacoes: {r.metadata['original_ops']} -> "
                  f"{r.metadata['fused_ops']} apos fusao")
    print(f"  sem fusao: {times[False]*1000:.1f} ms | "
          f"com fusao: {times[True]*1000:.1f} ms | "
          f"speedup: {times[False]/times[True]:.2f}x")


def bench_diagonal_path() -> None:
    print("\nCaminho diagonal (200 portas rz/cz em 20 qubits):")
    rng = np.random.default_rng(3)
    sim = StatevectorSimulator(20, seed=0)
    for q in range(20):
        sim.apply_gate("h", [q])
    t0 = time.perf_counter()
    for _ in range(200):
        if rng.random() < 0.7:
            sim.apply_gate("rz", [int(rng.integers(0, 20))],
                           [float(rng.random())])
        else:
            a, b = rng.choice(20, size=2, replace=False)
            sim.apply_gate("cz", [int(a), int(b)])
    dt = time.perf_counter() - t0
    print(f"  {200 / dt:.0f} portas diagonais/s ({1000 * dt / 200:.3f} ms/porta)")


def bench_sampling() -> None:
    print("\nAmostragem vetorizada (estado de 20 qubits totalmente emaranhado):")
    sim = StatevectorSimulator(20, seed=0)
    for q in range(20):
        sim.apply_gate("h", [q])
    for shots in (10_000, 100_000, 1_000_000):
        t0 = time.perf_counter()
        counts = sim.sample(shots)
        dt = time.perf_counter() - t0
        assert sum(counts.values()) == shots
        print(f"  {shots:>9} shots em {dt*1000:>8.1f} ms "
              f"({shots/dt/1e6:.2f} M shots/s)")


def bench_verify_overhead() -> None:
    print("\nCusto da verificacao cruzada c64/c128 (12 qubits, 150 portas):")
    ops = _random_ops(12, 150, seed=4)
    base = {"num_qubits": 12, "shots": 1024, "seed": 0,
            "precision": "complex64", "operations": ops}
    t0 = time.perf_counter()
    run_circuit(dict(base))
    t_plain = time.perf_counter() - t0
    t0 = time.perf_counter()
    r = run_circuit({**base, "verify": True})
    t_verify = time.perf_counter() - t0
    v = r.metadata["verification"]
    print(f"  sem verify: {t_plain*1000:.1f} ms | com verify: "
          f"{t_verify*1000:.1f} ms ({t_verify/t_plain:.2f}x)")
    print(f"  fidelidade: {v['state_fidelity']:.9f} | TVD: "
          f"{v['total_variation_distance']:.2e} | passed: {v['passed']}")


if __name__ == "__main__":
    print(f"Backends disponiveis neste host: {available_backends()}\n")
    bench_gates_per_second()
    bench_fusion()
    bench_diagonal_path()
    bench_sampling()
    bench_verify_overhead()
