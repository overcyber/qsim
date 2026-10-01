"""Suite de validacao em GPU e aceleradores experimentais.

Execute NA MAQUINA COM AS PLACAS (3090 / 4060 Ti / P40):

    pip install cupy-cuda12x                      # todas as placas
    pip install cuquantum-python-cu12             # so 3090/4060 Ti (CC >= 7.0)
    pip install numba                             # acelerador CPU opcional

    PYTHONPATH=. python3 tests/test_gpu.py                 # tudo que der
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 tests/test_gpu.py   # por placa

Cada bloco pula com aviso claro quando o requisito nao esta presente, entao
o script e seguro de rodar em qualquer maquina. Criterio de aceitacao dos
modulos experimentais (custatevec e numba): fidelidade > 1 - 1e-10 contra o
caminho de referencia nos circuitos aleatorios abaixo.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engine.backends import available_backends, to_cpu
from services.engine.simulator import make_simulator, run_circuit

GATES_1Q = ["h", "x", "y", "z", "s", "t", "sx"]
GATES_1Q_P = ["rx", "ry", "rz", "p"]
GATES_2Q = ["cx", "cz", "swap"]


def random_ops(rng, n: int, depth: int) -> list[dict]:
    ops = []
    for _ in range(depth):
        r = rng.random()
        if r < 0.45:
            ops.append({"gate": GATES_1Q[rng.integers(len(GATES_1Q))],
                        "targets": [int(rng.integers(n))]})
        elif r < 0.7:
            ops.append({"gate": GATES_1Q_P[rng.integers(len(GATES_1Q_P))],
                        "targets": [int(rng.integers(n))],
                        "params": [float(rng.uniform(-np.pi, np.pi))]})
        else:
            a, b = rng.choice(n, size=2, replace=False)
            ops.append({"gate": GATES_2Q[rng.integers(len(GATES_2Q))],
                        "targets": [int(a), int(b)]})
    return ops


def state_of(backend: str, n: int, ops: list[dict],
             precision: str = "complex128") -> np.ndarray:
    sim = make_simulator(n, seed=0, precision=precision, backend=backend)
    for op in ops:
        sim.apply_gate(op["gate"], op["targets"], op.get("params"))
    return to_cpu(sim.state).astype(np.complex128)


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.vdot(a, b)) ** 2)


# ---------------------------------------------------------------------------
def test_cupy_vs_numpy(trials: int = 10) -> None:
    if "cupy" not in available_backends():
        print("[pulado] cupy indisponivel nesta maquina")
        return
    rng = np.random.default_rng(1)
    worst = 1.0
    for _ in range(trials):
        n = int(rng.integers(4, 16))
        ops = random_ops(rng, n, int(rng.integers(20, 80)))
        f = fidelity(state_of("numpy", n, ops), state_of("cupy", n, ops))
        worst = min(worst, f)
        assert f > 1 - 1e-10, f
    print(f"[ok] cupy vs numpy: {trials} circuitos, pior fidelidade {worst:.12f}")


def test_custatevec_vs_cupy(trials: int = 10) -> None:
    if "custatevec" not in available_backends():
        print("[pulado] custatevec indisponivel (requer cuquantum-python e "
              "GPU CC >= 7.0; na P40 e esperado pular)")
        return
    rng = np.random.default_rng(2)
    worst = 1.0
    for _ in range(trials):
        n = int(rng.integers(4, 16))
        ops = random_ops(rng, n, int(rng.integers(20, 80)))
        f = fidelity(state_of("cupy", n, ops), state_of("custatevec", n, ops))
        worst = min(worst, f)
        assert f > 1 - 1e-10, \
            f"custatevec divergiu (fidelidade {f}) - checar convencao de alvos"
    print(f"[ok] custatevec vs cupy: {trials} circuitos, "
          f"pior fidelidade {worst:.12f}")
    # ordem de alvos em portas de 2-3 qubits (CX, CCX assimetricas)
    ops = [{"gate": "h", "targets": [0]}, {"gate": "cx", "targets": [0, 2]},
           {"gate": "ccx", "targets": [0, 2, 1]}]
    f = fidelity(state_of("cupy", 3, ops), state_of("custatevec", 3, ops))
    assert f > 1 - 1e-10, f"convencao de alvos incorreta (fidelidade {f})"
    print("[ok] custatevec: convencao de alvos CX/CCX confirmada")


def test_numba_vs_numpy(trials: int = 6) -> None:
    try:
        from services.engine.numba_kernels import HAS_NUMBA
    except Exception:
        HAS_NUMBA = False
    if not HAS_NUMBA:
        print("[pulado] numba nao instalado")
        return
    os.environ["QSIM_USE_NUMBA"] = "1"
    os.environ["QSIM_NUMBA_MIN_QUBITS"] = "2"
    import importlib
    from services.engine import simulator as simod
    importlib.reload(simod)
    rng = np.random.default_rng(3)
    worst = 1.0
    for _ in range(trials):
        n = int(rng.integers(4, 14))
        ops = random_ops(rng, n, int(rng.integers(20, 60)))
        ref = state_of("numpy", n, ops)  # modulo original, sem numba
        sim = simod.make_simulator(n, seed=0, backend="numpy")
        assert sim._numba, "numba nao ativou apos reload"
        for op in ops:
            sim.apply_gate(op["gate"], op["targets"], op.get("params"))
        f = fidelity(ref, to_cpu(sim.state).astype(np.complex128))
        worst = min(worst, f)
        assert f > 1 - 1e-10, f
    os.environ["QSIM_USE_NUMBA"] = "0"
    importlib.reload(simod)
    print(f"[ok] numba vs numpy: {trials} circuitos, pior fidelidade {worst:.12f}")


# ---------------------------------------------------------------------------
def bench(backend: str, n: int, depth: int = 60,
          precision: str = "complex64") -> float | None:
    if backend.split("-")[0] not in available_backends():
        return None
    rng = np.random.default_rng(7)
    ops = random_ops(rng, n, depth)
    try:
        sim = make_simulator(n, seed=0, precision=precision, backend=backend)
    except Exception as err:
        print(f"    {backend}: indisponivel ({err})")
        return None
    for op in ops[:5]:
        sim.apply_gate(op["gate"], op["targets"], op.get("params"))  # aquecer
    if hasattr(sim.xp, "cuda"):
        sim.xp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for op in ops:
        sim.apply_gate(op["gate"], op["targets"], op.get("params"))
    if hasattr(sim.xp, "cuda"):
        sim.xp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0
    return depth / dt


def benchmark_backends() -> None:
    print("\nBenchmark portas/s (complex64, circuito aleatorio, profundidade 60):")
    sizes = [16, 20, 24]
    if "cupy" in available_backends():
        sizes += [26, 28]
    for n in sizes:
        line = f"  {n:2d} qubits:"
        for backend in ("numpy", "cupy", "custatevec"):
            rate = bench(backend, n)
            if rate is not None:
                line += f"  {backend}={rate:8.1f}"
        print(line)
    if "cupy" in available_backends():
        import cupy as cp
        dev = cp.cuda.Device()
        free, total = dev.mem_info
        cc = int(dev.compute_capability)
        max31 = "sim" if total >= 17 * 2**30 else "nao (VRAM < 17 GiB)"
        print(f"\nGPU ativa: CC {cc / 10:.1f}, VRAM {total / 2**30:.1f} GiB "
              f"(livre {free / 2**30:.1f}) - 31 qubits c64: {max31}")


def test_31_qubits_smoke() -> None:
    """Opcional e pesado (16 GiB de VRAM): QSIM_GPU_SMOKE_31=1 para ativar."""
    if os.environ.get("QSIM_GPU_SMOKE_31") != "1":
        print("[pulado] smoke de 31 qubits (ative com QSIM_GPU_SMOKE_31=1)")
        return
    if "cupy" not in available_backends():
        print("[pulado] smoke de 31 qubits requer GPU")
        return
    t0 = time.perf_counter()
    result = run_circuit({
        "num_qubits": 31, "shots": 1000, "precision": "complex64",
        "backend": "cupy", "seed": 0,
        "operations": [{"gate": "h", "targets": [0]}] +
                      [{"gate": "cx", "targets": [i, i + 1]}
                       for i in range(30)],
    })
    dt = time.perf_counter() - t0
    assert set(result.counts.keys()) <= {"0" * 31, "1" * 31}
    print(f"[ok] GHZ de 31 qubits em GPU: {dt:.1f}s, "
          f"counts {result.counts}")


if __name__ == "__main__":
    print(f"Backends detectados: {available_backends()}\n")
    test_cupy_vs_numpy()
    test_custatevec_vs_cupy()
    test_numba_vs_numpy()
    test_31_qubits_smoke()
    benchmark_backends()
    print("\nValidacao GPU concluida (blocos pulados indicam requisito ausente).")
