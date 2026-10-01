"""Suite fisica (oraculo): resultados com valor analitico conhecido.

Todo backend ou otimizacao nova precisa passar aqui antes de entrar.
Executavel direto (python tests/test_physics.py) ou via pytest.
"""
import time

import numpy as np

from services.engine.simulator import StatevectorSimulator, run_circuit


def test_bell_state():
    result = run_circuit({
        "num_qubits": 2, "shots": 10000, "seed": 42,
        "operations": [
            {"gate": "h", "targets": [0]},
            {"gate": "cx", "targets": [0, 1]},
        ],
    })
    assert set(result.counts.keys()) == {"00", "11"}
    assert abs(result.counts["00"] / result.shots - 0.5) < 0.03
    print("Bell state OK:", result.counts)


def test_ghz_20_qubits():
    ops = [{"gate": "h", "targets": [0]}]
    ops += [{"gate": "cx", "targets": [i, i + 1]} for i in range(19)]
    t0 = time.perf_counter()
    result = run_circuit({"num_qubits": 20, "shots": 4096, "seed": 7,
                          "operations": ops})
    dt = (time.perf_counter() - t0) * 1000
    assert set(result.counts.keys()) == {"0" * 20, "1" * 20}
    print(f"GHZ 20 qubits OK em {dt:.1f} ms:", result.counts)


def test_ghz_20_qubits_c64():
    ops = [{"gate": "h", "targets": [0]}]
    ops += [{"gate": "cx", "targets": [i, i + 1]} for i in range(19)]
    result = run_circuit({"num_qubits": 20, "shots": 4096, "seed": 7,
                          "precision": "complex64", "operations": ops})
    assert set(result.counts.keys()) == {"0" * 20, "1" * 20}
    print("GHZ 20 qubits em complex64 OK")


def test_unitarity_random_circuit():
    rng = np.random.default_rng(0)
    sim = StatevectorSimulator(10, seed=0)
    names = ["h", "x", "y", "z", "s", "t", "sx"]
    for _ in range(200):
        if rng.random() < 0.7:
            sim.apply_gate(str(rng.choice(names)), [int(rng.integers(0, 10))])
        else:
            a, b = rng.choice(10, size=2, replace=False)
            sim.apply_gate("cx", [int(a), int(b)])
    norm = np.linalg.norm(sim.state)
    assert abs(norm - 1.0) < 1e-10
    print(f"Unitariedade OK: norma = {norm:.12f} apos 200 portas")


def test_parametric_rotation():
    result = run_circuit({
        "num_qubits": 1, "shots": 1, "seed": 1, "observables": ["z0"],
        "operations": [{"gate": "rx", "targets": [0], "params": [np.pi / 3]}],
    })
    expected = float(np.cos(np.pi / 3))
    assert abs(result.expectation["z0"] - expected) < 1e-10
    print(f"RX(pi/3) OK: <Z> = {result.expectation['z0']:.6f}")


def test_mid_circuit_measure():
    result = run_circuit({
        "num_qubits": 2, "shots": 500, "seed": 3,
        "operations": [
            {"gate": "h", "targets": [0]},
            {"op": "measure", "targets": [0]},
            {"gate": "cx", "targets": [0, 1]},
        ],
    })
    assert set(result.counts.keys()) <= {"00", "11"}
    print("Medicao intermediaria OK:", result.counts)


def test_noise_depolarizing():
    result = run_circuit({
        "num_qubits": 2, "shots": 300, "seed": 5,
        "noise": {"depolarizing": 0.05, "readout_error": 0.01},
        "operations": [
            {"gate": "h", "targets": [0]},
            {"gate": "cx", "targets": [0, 1]},
        ],
    })
    assert sum(result.counts.values()) == 300
    impure = sum(v for k, v in result.counts.items() if k not in ("00", "11"))
    assert impure > 0
    print("Ruido OK:", result.counts)


def test_qft_8_qubits():
    n = 8
    ops = [{"gate": "x", "targets": [0]}]
    for j in reversed(range(n)):
        ops.append({"gate": "h", "targets": [j]})
        for k in reversed(range(j)):
            ops.append({"gate": "cp", "targets": [k, j],
                        "params": [np.pi / (2 ** (j - k))]})
    result = run_circuit({"num_qubits": n, "shots": 1, "seed": 0,
                          "operations": ops, "return_statevector": True})
    sv = np.array([complex(re, im) for re, im in result.statevector])
    assert np.allclose(np.abs(sv), 1 / np.sqrt(2**n), atol=1e-10)
    print("QFT 8 qubits OK: amplitudes uniformes")


def test_fusion_preserves_semantics():
    """Circuito denso em portas de 1 qubit: fundido == nao fundido."""
    rng = np.random.default_rng(11)
    ops = []
    for _ in range(120):
        if rng.random() < 0.8:
            name = str(rng.choice(["h", "t", "s", "x", "rz"]))
            op = {"gate": name, "targets": [int(rng.integers(0, 6))]}
            if name == "rz":
                op["params"] = [float(rng.random())]
            ops.append(op)
        else:
            a, b = rng.choice(6, size=2, replace=False)
            ops.append({"gate": "cx", "targets": [int(a), int(b)]})
    base = {"num_qubits": 6, "shots": 1, "seed": 9, "operations": ops,
            "return_statevector": True}
    r_fused = run_circuit({**base, "fuse": True})
    r_plain = run_circuit({**base, "fuse": False})
    a = np.array([complex(x, y) for x, y in r_fused.statevector])
    b = np.array([complex(x, y) for x, y in r_plain.statevector])
    fid = abs(np.vdot(a, b)) ** 2
    assert fid > 1 - 1e-10, f"fidelidade fundido vs plano = {fid}"
    reduction = r_fused.metadata["fused_ops"] / r_fused.metadata["original_ops"]
    print(f"Fusao OK: fidelidade {fid:.12f}, "
          f"{r_fused.metadata['original_ops']} -> "
          f"{r_fused.metadata['fused_ops']} ops ({reduction:.0%})")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nSuite fisica: todos os testes passaram.")
