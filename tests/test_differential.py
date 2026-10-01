"""Suite diferencial: compara execucoes independentes que devem coincidir.

  1. complex64 vs complex128 (a base da politica "c64 padrao em GPU com
     verificacao cruzada em c128"): mede fidelidade de estado e distancia
     de variacao total em circuitos aleatorios profundos e verifica que o
     mecanismo verify=true integrado reporta os mesmos numeros.
  2. NumPy vs CuPy (executa apenas se houver GPU; e o teste de aceitacao
     de qualquer backend novo).
"""
import numpy as np

from services.engine.backends import available_backends
from services.engine.simulator import run_circuit


def _random_circuit(n: int, depth: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    ops = []
    for _ in range(depth):
        r = rng.random()
        if r < 0.5:
            ops.append({"gate": str(rng.choice(["h", "t", "s", "sx"])),
                        "targets": [int(rng.integers(0, n))]})
        elif r < 0.75:
            ops.append({"gate": str(rng.choice(["rx", "ry", "rz"])),
                        "targets": [int(rng.integers(0, n))],
                        "params": [float(rng.uniform(0, 2 * np.pi))]})
        else:
            a, b = rng.choice(n, size=2, replace=False)
            ops.append({"gate": str(rng.choice(["cx", "cz"])),
                        "targets": [int(a), int(b)]})
    return ops


def test_c64_vs_c128_deep_random_circuits():
    """Fidelidade c64/c128 em circuitos de 400 portas deve ficar > 1-1e-6."""
    worst_fid, worst_tvd = 1.0, 0.0
    for seed in range(5):
        ops = _random_circuit(n=10, depth=400, seed=seed)
        result = run_circuit({
            "num_qubits": 10, "shots": 1, "seed": seed,
            "precision": "complex64", "verify": True, "operations": ops,
        })
        v = result.metadata["verification"]
        assert v["passed"], f"verificacao falhou no seed {seed}: {v}"
        worst_fid = min(worst_fid, v["state_fidelity"])
        worst_tvd = max(worst_tvd, v["total_variation_distance"])
    print(f"Diferencial c64/c128 OK em 5 circuitos de 400 portas: "
          f"pior fidelidade {worst_fid:.9f}, pior TVD {worst_tvd:.2e}")


def test_verify_report_consistency():
    """verify=true partindo de c128 compara contra c64 e vice-versa."""
    ops = _random_circuit(n=6, depth=100, seed=42)
    r64 = run_circuit({"num_qubits": 6, "shots": 1, "seed": 1,
                       "precision": "complex64", "verify": True,
                       "operations": ops})
    r128 = run_circuit({"num_qubits": 6, "shots": 1, "seed": 1,
                        "precision": "complex128", "verify": True,
                        "operations": ops})
    v64, v128 = r64.metadata["verification"], r128.metadata["verification"]
    assert v64["compared"] == "complex64 vs complex128"
    assert v128["compared"] == "complex128 vs complex64"
    diff = abs(v64["state_fidelity"] - v128["state_fidelity"])
    assert diff < 1e-9, f"relatorios divergem: {diff}"
    print(f"Relatorio de verificacao simetrico OK (delta {diff:.2e})")


def test_same_seed_same_counts():
    """Reprodutibilidade: mesma seed -> mesmas contagens, sempre."""
    ops = _random_circuit(n=8, depth=60, seed=7)
    spec = {"num_qubits": 8, "shots": 2048, "seed": 123, "operations": ops}
    r1, r2 = run_circuit(dict(spec)), run_circuit(dict(spec))
    assert r1.counts == r2.counts
    print("Reprodutibilidade por seed OK")


def test_numpy_vs_cupy_if_available():
    if "cupy" not in available_backends():
        print("CuPy indisponivel neste host: teste NumPy vs CuPy pulado "
              "(rode na maquina com as GPUs)")
        return
    ops = _random_circuit(n=12, depth=200, seed=3)
    base = {"num_qubits": 12, "shots": 1, "seed": 3,
            "return_statevector": True, "operations": ops}
    r_cpu = run_circuit({**base, "backend": "numpy"})
    r_gpu = run_circuit({**base, "backend": "cupy"})
    a = np.array([complex(x, y) for x, y in r_cpu.statevector])
    b = np.array([complex(x, y) for x, y in r_gpu.statevector])
    fid = abs(np.vdot(a, b)) ** 2
    assert fid > 1 - 1e-9, f"fidelidade numpy/cupy = {fid}"
    print(f"Diferencial NumPy vs CuPy OK: fidelidade {fid:.12f}")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nSuite diferencial: todos os testes passaram.")
