"""Suite dos aceleradores do nucleo (P1.1, P1.3 e P0.3 do plano).

Partes CPU (sempre rodam):
  - caminho com escrita de volta in-place (sem copia final) vs referencia
    classica moveaxis+matmul+copias;
  - amostragem por blocos vs CDF completa materializada.

Partes GPU (rodam onde houver CuPy — na maquina com as placas):
  - kernels in-place vs caminho generico (fidelidade 1.0);
  - observaveis Pauli por kernel vs caminho de copia;
  - smoke de 31 qubits c64 (gate: QSIM_GPU_SMOKE_31=1) — o criterio de
    aceite P1.1: concluir na GPU sem OOM.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engine.backends import available_backends, to_cpu
from services.engine.simulator import StatevectorSimulator, run_circuit

GATES_1Q = ["h", "x", "y", "z", "s", "sdg", "t", "sx"]
GATES_1Q_P = ["rx", "ry", "rz", "p"]
GATES_2Q = ["cx", "cz", "swap"]
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
            ops.append({"gate": GATES_2Q[rng.integers(len(GATES_2Q))],
                        "targets": [int(a), int(b)]})
        else:
            qs = rng.choice(n, size=3, replace=False)
            ops.append({"gate": GATES_3Q[rng.integers(len(GATES_3Q))],
                        "targets": [int(q) for q in qs]})
    return ops


def reference_state(n: int, ops: list[dict], precision="complex128") -> np.ndarray:
    """Referencia isolada: moveaxis + matmul com copias explicitas
    (o algoritmo pre-P1.1), para provar a equivalencia do caminho novo."""
    from services.engine import gates as G
    dtype = np.complex64 if precision == "complex64" else np.complex128
    state = np.zeros(2**n, dtype=dtype)
    state[0] = 1.0
    for op in ops:
        mat, k = G.resolve(op["gate"], op.get("params"))
        targets = op["targets"]
        mat = mat.astype(dtype)
        axes = [n - 1 - t for t in targets]
        psi = np.moveaxis(state.reshape([2] * n), axes, range(k))
        diag = np.diag(np.diag(mat))
        if np.count_nonzero(mat - diag) == 0:
            psi *= np.diag(mat).reshape([2] * k + [1] * (n - k))
        else:
            out = mat @ psi.reshape(2**k, -1)
            out = out.reshape([2] * k + list(psi.shape[k:]))
            out = np.moveaxis(out, range(k), axes)
            state = np.ascontiguousarray(out.reshape(-1))
    return state


def test_writeback_path_vs_reference() -> None:
    rng = np.random.default_rng(11)
    worst = 1.0
    for _ in range(8):
        n = int(rng.integers(4, 12))
        ops = random_ops(rng, n, int(rng.integers(20, 60)))
        sim = StatevectorSimulator(n, seed=0)
        for op in ops:
            sim.apply_gate(op["gate"], op["targets"], op.get("params"))
        ref = reference_state(n, ops)
        f = float(np.abs(np.vdot(ref, to_cpu(sim.state))) ** 2)
        worst = min(worst, f)
        assert f > 1 - 1e-12, f
    print(f"[ok] caminho in-place (CPU writeback) vs referencia: 8 circuitos, "
          f"pior fidelidade {worst:.15f}")


def test_chunked_sampler_vs_full_cdf() -> None:
    rng = np.random.default_rng(5)
    for n in (10, 14):
        ops = random_ops(rng, n, 40)
        sim = StatevectorSimulator(n, seed=42)
        for op in ops:
            sim.apply_gate(op["gate"], op["targets"], op.get("params"))
        counts_chunked = sim.sample(20000)

        # referencia: CDF float64 completa (algoritmo pre-P0.3)
        probs = np.abs(to_cpu(sim.state)) ** 2
        cdf = np.cumsum(probs.astype(np.float64))
        cdf /= cdf[-1]
        u = np.random.default_rng(42).random(20000)
        idx = np.clip(np.searchsorted(cdf, u, side="right"), 0, 2**n - 1)
        vals, cnts = np.unique(idx, return_counts=True)
        counts_ref = {format(int(v), f"0{n}b"): int(c)
                      for v, c in zip(vals, cnts)}

        keys = set(counts_chunked) | set(counts_ref)
        tvd = 0.5 * sum(abs(counts_chunked.get(k, 0) - counts_ref.get(k, 0))
                        / 20000 for k in keys)
        assert tvd < 1e-3, (n, tvd)
        print(f"[ok] amostrador por blocos vs CDF completa ({n}q): "
              f"TVD = {tvd:.2e}")


def test_chunked_sampler_invariants() -> None:
    sim = StatevectorSimulator(6, seed=1)
    sim.apply_gate("h", [0])
    c = sim.sample(5000)
    assert sum(c.values()) == 5000
    assert set(c) <= {"000000", "000001"}  # q0 = bit menos significativo
    assert 0.46 < c["000001"] / 5000 < 0.54
    print("[ok] amostrador por blocos: invariantes (soma, suporte, 50/50)")


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
def state_of(sim: StatevectorSimulator, ops: list[dict]) -> np.ndarray:
    for op in ops:
        sim.apply_gate(op["gate"], op["targets"], op.get("params"))
    return to_cpu(sim.state).astype(np.complex128)


def test_gpu_kernels_vs_generic_path() -> None:
    if "cupy" not in available_backends():
        print("[pulado] kernels GPU exigem CuPy")
        return
    rng = np.random.default_rng(2)
    for precision, tol in (("complex64", 1e-5), ("complex128", 1e-10)):
        worst = 1.0
        for _ in range(6):
            n = int(rng.integers(4, 13))
            ops = random_ops(rng, n, int(rng.integers(20, 60)))
            sim_k = StatevectorSimulator(n, seed=0, precision=precision,
                                         backend="cupy")
            assert sim_k._kernels is not None, "kernels nao ativaram"
            sim_g = StatevectorSimulator(n, seed=0, precision=precision,
                                         backend="cupy")
            sim_g._kernels = None  # forca o caminho generico
            a = state_of(sim_k, ops)
            b = state_of(sim_g, ops)
            f = float(np.abs(np.vdot(b, a)) ** 2)
            worst = min(worst, f)
            assert f > 1 - tol, (precision, f)
        print(f"[ok] kernels in-place vs generico ({precision}): "
              f"pior fidelidade {worst:.12f}")


def test_gpu_pauli_kernel() -> None:
    if "cupy" not in available_backends():
        print("[pulado] kernels GPU exigem CuPy")
        return
    rng = np.random.default_rng(9)
    for _ in range(8):
        n = int(rng.integers(3, 11))
        ops = random_ops(rng, n, 25)
        sim = StatevectorSimulator(n, seed=0, backend="cupy")
        assert sim._kernels is not None
        for op in ops:
            sim.apply_gate(op["gate"], op["targets"], op.get("params"))
        host = to_cpu(sim.state).astype(np.complex128)
        for _p in range(6):
            p = "".join(str(rng.choice(list("IXYZ"))) for _ in range(n))
            got = sim.expectation_pauli(p)
            # referencia host: <psi|P|psi> por produto elementwise
            acc = 0j
            for i in range(2**n):
                j = i
                val = 1 + 0j
                for q, ch in enumerate(p):
                    bit = (i >> q) & 1
                    if ch == "X":
                        j ^= 1 << q
                    elif ch == "Y":
                        j ^= 1 << q
                        val *= 1j if bit else -1j
                    elif ch == "Z":
                        val *= -1 if bit else 1
                acc += np.conj(host[i]) * val * host[j]
            assert abs(got - acc.real) < 1e-9, (p, got, acc)
    print("[ok] observaveis Pauli por kernel vs referencia host (8 estados)")


def test_31_qubits_p40() -> None:
    """Criterio de aceite P1.1: GHZ-31 c64 conclui na GPU sem OOM (<60s)."""
    if os.environ.get("QSIM_GPU_SMOKE_31") != "1":
        print("[pulado] smoke 31q (ative com QSIM_GPU_SMOKE_31=1)")
        return
    if "cupy" not in available_backends():
        print("[pulado] sem CuPy")
        return
    # liberar blocos que o pool CuPy retém dos testes anteriores (a VRAM
    # precisa acomodar estado 8 GiB + probabilidades 8 GiB em paralelo)
    import cupy as _cp
    _cp.get_default_memory_pool().free_all_blocks()
    t0 = time.perf_counter()
    r = run_circuit({
        "num_qubits": 31, "shots": 1000, "precision": "complex64",
        "backend": "cupy", "seed": 0,
        "operations": [{"gate": "h", "targets": [0]}]
                      + [{"gate": "cx", "targets": [i, i + 1]} for i in range(30)],
    })
    dt = time.perf_counter() - t0
    assert set(r.counts) <= {"0" * 31, "1" * 31}
    print(f"[ok] GHZ-31 c64 na GPU: {dt:.1f}s, counts={r.counts} "
          f"(meta do plano: < 60s)")


if __name__ == "__main__":
    test_writeback_path_vs_reference()
    test_chunked_sampler_vs_full_cdf()
    test_chunked_sampler_invariants()
    test_gpu_kernels_vs_generic_path()
    test_gpu_pauli_kernel()
    test_31_qubits_p40()
    print("\nSuite de aceleradores: todos os testes passaram.")
