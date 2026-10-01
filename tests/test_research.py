"""Suite de recursos de pesquisa: QASM, observaveis Pauli, Bloch,
entropia de emaranhamento, ruido estendido, trajetorias paralelas e
cancelamento cooperativo.
"""
import math
import time

import numpy as np

from services.engine.qasm import QasmError, parse
from services.engine.simulator import JobCancelled, run_circuit
from shared.validation import validate_circuit


# ---------------------------------------------------------------------------
# QASM
# ---------------------------------------------------------------------------
def test_qasm_bell_qasm3_and_qasm2():
    for src in [
        "OPENQASM 3; qubit[2] q; h q[0]; cx q[0], q[1];",
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
        "h q[0]; cx q[0],q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];",
    ]:
        circuit = parse(src)
        assert circuit["num_qubits"] == 2
        result = run_circuit({**circuit, "shots": 2000, "seed": 1})
        assert set(result.counts.keys()) <= {"00", "11"}
    print("QASM Bell (versoes 2 e 3) OK")


def test_qasm_params_and_broadcast():
    circuit = parse("""
        OPENQASM 3;
        qubit[3] q;
        h q;                 // broadcast
        rz(pi/4) q[1];
        cp(3*pi/4) q[0], q[2];
        barrier;
        measure q;
    """)
    assert circuit["num_qubits"] == 3
    kinds = [op.get("op", "gate") for op in circuit["operations"]]
    meas = [o for o in circuit["operations"] if o.get("op") == "measure"]
    assert sum(len(m["targets"]) for m in meas) == 3 and "barrier" in kinds
    rz = next(o for o in circuit["operations"] if o.get("gate") == "rz")
    assert abs(rz["params"][0] - math.pi / 4) < 1e-12
    run_circuit({**circuit, "shots": 10, "seed": 0})
    print("QASM parametros com pi e broadcast OK")


def test_qasm_hostile_inputs_rejected():
    hostile = [
        "",
        "OPENQASM 3; qubit[2] q; h q[5];",
        "OPENQASM 3; qubit[99] q; h q[0];",
        "OPENQASM 3; qubit[2] q; rz(__import__) q[0];",
        "OPENQASM 3; qubit[2] q; rz(9**9**9) q[0];",
        "OPENQASM 3; qubit[2] q; qubit[2] r; h q[0];",
        "h q[0];",
    ]
    for src in hostile:
        try:
            parse(src)
        except (QasmError, ValueError):
            continue
        raise AssertionError(f"QASM hostil aceito: {src[:50]}")
    print("QASM: entradas hostis rejeitadas OK")


# ---------------------------------------------------------------------------
# Observaveis Pauli arbitrarios
# ---------------------------------------------------------------------------
def test_pauli_observables_bell():
    """No estado de Bell: <ZZ> = <XX> = 1, <ZI> = <IZ> = 0."""
    result = run_circuit({
        "num_qubits": 2, "shots": 1, "seed": 0,
        "observables": ["ZZ", "XX", "ZI", "IZ", "z0"],
        "operations": [{"gate": "h", "targets": [0]},
                       {"gate": "cx", "targets": [0, 1]}],
    })
    e = result.expectation
    assert abs(e["ZZ"] - 1.0) < 1e-10
    assert abs(e["XX"] - 1.0) < 1e-10
    assert abs(e["ZI"]) < 1e-10 and abs(e["IZ"]) < 1e-10
    assert abs(e["z0"]) < 1e-10
    print("Observaveis Pauli no Bell OK:",
          {k: round(v, 6) for k, v in e.items()})


def test_pauli_matches_z_shorthand():
    ops = [{"gate": "ry", "targets": [0], "params": [0.9]},
           {"gate": "cx", "targets": [0, 1]}]
    r = run_circuit({"num_qubits": 2, "shots": 1, "seed": 0,
                     "observables": ["z1", "IZ"], "operations": ops})
    assert abs(r.expectation["z1"] - r.expectation["IZ"]) < 1e-12
    print("Pauli IZ == z1 OK")


# ---------------------------------------------------------------------------
# Bloch e entropia de emaranhamento
# ---------------------------------------------------------------------------
def test_bloch_vectors():
    r = run_circuit({
        "num_qubits": 3, "shots": 1, "seed": 0, "return_bloch": True,
        "operations": [
            {"gate": "x", "targets": [0]},          # |1>: z = -1
            {"gate": "h", "targets": [1]},          # |+>: x = +1
            {"gate": "h", "targets": [2]},
            {"gate": "s", "targets": [2]},          # |+i>: y = +1
        ],
    })
    b = r.bloch
    assert abs(b["0"][2] + 1.0) < 1e-9
    assert abs(b["1"][0] - 1.0) < 1e-9
    assert abs(b["2"][1] - 1.0) < 1e-9
    print("Vetores de Bloch OK:", b)


def test_entanglement_entropy_ghz_vs_product():
    ghz = run_circuit({
        "num_qubits": 3, "shots": 1, "seed": 0, "return_entanglement": True,
        "operations": [{"gate": "h", "targets": [0]},
                       {"gate": "cx", "targets": [0, 1]},
                       {"gate": "cx", "targets": [1, 2]}],
    })
    prod = run_circuit({
        "num_qubits": 3, "shots": 1, "seed": 0, "return_entanglement": True,
        "operations": [{"gate": "h", "targets": [0]},
                       {"gate": "h", "targets": [1]},
                       {"gate": "h", "targets": [2]}],
    })
    assert all(abs(s - 1.0) < 1e-9 for s in ghz.entanglement.values())
    assert all(s < 1e-9 for s in prod.entanglement.values())
    print("Entropia: GHZ = 1 bit/qubit, produto = 0 OK")


# ---------------------------------------------------------------------------
# Ruido por porta e por qubit
# ---------------------------------------------------------------------------
def test_noise_per_gate_and_per_qubit():
    circuit = {
        "num_qubits": 2, "shots": 400, "seed": 9, "parallel": False,
        "noise": {
            "per_gate": {"cx": {"depolarizing": 0.2}},
            "per_qubit": {"1": {"depolarizing": 0.1}},
        },
        "operations": [{"gate": "h", "targets": [0]},
                       {"gate": "cx", "targets": [0, 1]}],
    }
    validate_circuit(circuit)
    result = run_circuit(circuit)
    impure = sum(v for k, v in result.counts.items() if k not in ("00", "11"))
    assert impure > 0
    clean = run_circuit({**circuit, "noise": None})
    assert set(clean.counts.keys()) == {"00", "11"}
    print(f"Ruido por porta/qubit OK ({impure}/400 estados impuros)")


# ---------------------------------------------------------------------------
# Trajetorias paralelas
# ---------------------------------------------------------------------------
def test_parallel_trajectories_consistency():
    circuit = {
        "num_qubits": 4, "shots": 256, "seed": 5,
        "noise": {"depolarizing": 0.02},
        "operations": [{"gate": "h", "targets": [0]},
                       {"gate": "cx", "targets": [0, 1]},
                       {"gate": "cx", "targets": [1, 2]},
                       {"gate": "cx", "targets": [2, 3]}],
    }
    t0 = time.perf_counter()
    par = run_circuit({**circuit, "parallel": True})
    t_par = time.perf_counter() - t0
    seq = run_circuit({**circuit, "parallel": False})
    assert sum(par.counts.values()) == 256
    assert sum(seq.counts.values()) == 256
    assert par.metadata.get("parallel_workers", 1) >= 1
    dominant_par = max(par.counts, key=par.counts.get)
    assert dominant_par in ("0000", "1111")
    print(f"Trajetorias paralelas OK: {par.metadata['parallel_workers']} "
          f"workers, {t_par*1000:.0f} ms")


# ---------------------------------------------------------------------------
# Cancelamento cooperativo
# ---------------------------------------------------------------------------
def test_cancellation():
    ops = [{"gate": "h", "targets": [i % 18]} for i in range(2000)]
    calls = {"n": 0}

    def cancel_after_few() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    try:
        run_circuit({"num_qubits": 18, "shots": 10, "fuse": False,
                     "operations": ops}, should_cancel=cancel_after_few)
    except JobCancelled:
        print(f"Cancelamento cooperativo OK (apos {calls['n']} verificacoes)")
        return
    raise AssertionError("Job nao foi cancelado")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("\nSuite de pesquisa: todos os testes passaram.")
