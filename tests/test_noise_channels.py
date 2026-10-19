"""Suite de canais de ruido e paridade entre motores.

Existe porque dois defeitos passaram pelas suites anteriores:
  1. o amplitude damping do nucleo era um canal de reset (coerencias
     decaiam por 1-gamma em vez de sqrt(1-gamma));
  2. o Aer ignorava ruido em rzz, rxx e id e falhava com apelidos
     (cnot, toffoli, fredkin, phase, i) e com iswap.
Cobre tambem o phase damping e o relaxamento termico (T1, T2, duracao
de porta), conferidos contra o canal do Qiskit e contra as curvas de
Ramsey e de decaimento de T1.
O teste de paridade antigo so cobria depolarizante e erro de leitura.

Politica desta suite: todo canal e conferido contra a formula analitica
(oraculo independente dos dois motores) e todo nome de porta aceito pela
validacao e conferido nos dois motores.

Uso: PYTHONPATH=. python3 tests/test_noise_channels.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.engine import gates
from services.engine.backend_aer import aer_available
from services.engine.simulator import (NoiseModel, run_circuit,
                                      thermal_relaxation_probs)
from services.scheduler.main import classify

SHOTS = 20000
HAS_AER = aer_available()
ENGINES = ["numpy", "aer"] if HAS_AER else ["numpy"]


def _sigma(p: float, shots: int = SHOTS) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / shots)


def _frac(counts: dict, pred, shots: int = SHOTS) -> float:
    return sum(v for k, v in counts.items() if pred(k)) / shots


def _check(nome: str, medido: float, esperado: float, motor: str) -> None:
    tol = 5 * _sigma(esperado)
    assert abs(medido - esperado) < tol, (
        f"{nome} [{motor}]: medido {medido:.4f}, esperado {esperado:.4f} "
        f"(tolerancia 5 sigma = {tol:.4f})"
    )


IDLE = {"gate": "rx", "targets": [0], "params": [0.0]}


# ---------------------------------------------------------------------------
# Canais contra formulas analiticas
# ---------------------------------------------------------------------------
def test_amplitude_damping_coherence() -> None:
    """|+> -> AD(gamma) -> H: P(0) = 1/2 + sqrt(1-gamma)/2 (coerencia)."""
    for gamma in (0.2, 0.5, 0.8):
        esperado = 0.5 + math.sqrt(1 - gamma) / 2
        for motor in ENGINES:
            r = run_circuit({
                "num_qubits": 1, "shots": SHOTS, "seed": 11,
                "backend": motor,
                "noise": {"per_gate": {"rx": {"amplitude_damping": gamma}}},
                "operations": [{"gate": "h", "targets": [0]}, IDLE,
                               {"gate": "h", "targets": [0]}],
            })
            _check(f"AD coerencia gamma={gamma}",
                   _frac(r.counts, lambda k: k[-1] == "0"), esperado, motor)
    print("[ok] amplitude damping: coerencias decaem por sqrt(1-gamma) "
          "nos dois motores")


def test_amplitude_damping_population() -> None:
    """|1> -> AD(gamma): P(1) = 1 - gamma (populacao)."""
    gamma = 0.3
    for motor in ENGINES:
        r = run_circuit({
            "num_qubits": 1, "shots": SHOTS, "seed": 12, "backend": motor,
            "noise": {"per_gate": {"rx": {"amplitude_damping": gamma}}},
            "operations": [{"gate": "x", "targets": [0]}, IDLE],
        })
        _check("AD populacao", _frac(r.counts, lambda k: k[-1] == "1"),
               1 - gamma, motor)
    print("[ok] amplitude damping: populacao p1 -> (1-gamma) p1")


def test_depolarizing_and_readout() -> None:
    """|0> -> dep(p): P(1) = 2p/3. Leitura sozinha: P(1) = r."""
    p, r_err = 0.3, 0.07
    for motor in ENGINES:
        r = run_circuit({
            "num_qubits": 1, "shots": SHOTS, "seed": 13, "backend": motor,
            "noise": {"per_gate": {"rx": {"depolarizing": p}}},
            "operations": [IDLE],
        })
        _check("depolarizante", _frac(r.counts, lambda k: k[-1] == "1"),
               2 * p / 3, motor)
        r = run_circuit({
            "num_qubits": 1, "shots": SHOTS, "seed": 14, "backend": motor,
            "noise": {"readout_error": r_err},
            "operations": [IDLE],
        })
        _check("erro de leitura", _frac(r.counts, lambda k: k[-1] == "1"),
               r_err, motor)
        if motor == "numpy":
            assert r.metadata["mode"] == "exact", r.metadata
    print("[ok] depolarizante (2p/3) e erro de leitura; leitura sozinha "
          "usa o caminho exato")


def test_phase_damping() -> None:
    """|+> -> PD(lam) -> H: P(0) = 1/2 + sqrt(1-lam)/2; |1> fica intacto."""
    for lam in (0.2, 0.6):
        esperado = 0.5 + math.sqrt(1 - lam) / 2
        for motor in ENGINES:
            r = run_circuit({
                "num_qubits": 1, "shots": SHOTS, "seed": 15, "backend": motor,
                "noise": {"per_gate": {"rx": {"phase_damping": lam}}},
                "operations": [{"gate": "h", "targets": [0]}, IDLE,
                               {"gate": "h", "targets": [0]}],
            })
            _check(f"PD coerencia lam={lam}",
                   _frac(r.counts, lambda k: k[-1] == "0"), esperado, motor)
    for motor in ENGINES:
        r = run_circuit({
            "num_qubits": 1, "shots": 2000, "seed": 16, "backend": motor,
            "noise": {"phase_damping": 0.7},
            "operations": [{"gate": "x", "targets": [0]}, IDLE],
        })
        assert r.counts == {"1": 2000}, (motor, r.counts)
    print("[ok] phase damping: coerencias por sqrt(1-lam), populacoes "
          "intactas, nos dois motores")


def test_thermal_relaxation_matches_qiskit() -> None:
    """AD(gamma) seguido de PD(lam) e o thermal_relaxation_error do Qiskit."""
    try:
        from qiskit.quantum_info import SuperOp
        from qiskit_aer.noise import (amplitude_damping_error,
                                      phase_damping_error,
                                      thermal_relaxation_error)
    except ImportError:
        print("[pulado] qiskit-aer ausente")
        return
    casos = ((100e-6, 80e-6, 1e-6), (100e-6, 200e-6, 5e-6),
             (50e-6, 10e-6, 3e-6), (300e-6, 299e-6, 50e-9))
    for t1, t2, tau in casos:
        gamma, lam = thermal_relaxation_probs(t1, t2, tau)
        nosso = SuperOp(amplitude_damping_error(gamma).to_quantumchannel()) \
            .compose(SuperOp(phase_damping_error(lam).to_quantumchannel()))
        ref = SuperOp(thermal_relaxation_error(t1, t2, tau).to_quantumchannel())
        dif = np.max(np.abs(nosso.data - ref.data))
        assert dif < 1e-12, (t1, t2, tau, dif)
    g, l = thermal_relaxation_probs(math.inf, 40e-6, 4e-6)
    assert g == 0.0 and abs(l - (1 - math.exp(-0.2))) < 1e-15
    g, l = thermal_relaxation_probs(40e-6, 80e-6, 4e-6)
    assert abs(g - (1 - math.exp(-0.1))) < 1e-15 and l < 1e-15
    print(f"[ok] relaxamento termico: {len(casos)} casos iguais ao canal do "
          "Qiskit (1e-12); limites T1 infinito e T2 = 2 T1")


def test_thermal_relaxation_curves() -> None:
    """Ramsey no qubit 0 e decaimento de T1 no qubit 1, com T1 e T2
    diferentes por qubit e duracao so na porta id."""
    k, tau = 5, 10e-6
    t1_q0, t2_q0, t1_q1 = 100e-6, 50e-6, 30e-6
    noise = {"per_gate": {"id": {"gate_time": tau}},
             "per_qubit": {"0": {"t1": t1_q0, "t2": t2_q0},
                           "1": {"t1": t1_q1}}}
    ops = [{"gate": "h", "targets": [0]}, {"gate": "x", "targets": [1]}]
    ops += [{"gate": "i" if j % 2 else "id", "targets": [q]}
            for j in range(k) for q in (0, 1)]
    ops += [{"gate": "h", "targets": [0]}]
    ramsey = 0.5 + 0.5 * math.exp(-k * tau / t2_q0)
    decaimento = math.exp(-k * tau / t1_q1)
    res = {}
    for motor in ENGINES:
        r = run_circuit({"num_qubits": 2, "shots": SHOTS, "seed": 17,
                         "backend": motor, "noise": noise, "operations": ops})
        res[motor] = r.counts
        _check("Ramsey q0", _frac(r.counts, lambda s: s[-1] == "0"), ramsey,
               motor)
        _check("T1 q1", _frac(r.counts, lambda s: s[-2] == "1"), decaimento,
               motor)
    if len(res) == 2:
        a, b = res["numpy"], res["aer"]
        tvd = 0.5 * sum(abs(a.get(s, 0) - b.get(s, 0))
                        for s in set(a) | set(b)) / SHOTS
        assert tvd < 0.03, f"relaxamento: TVD entre motores {tvd:.4f}"
    print(f"[ok] relaxamento termico: Ramsey (P0 = {ramsey:.3f}) e "
          f"decaimento de T1 (P1 = {decaimento:.3f}) nos dois motores")


def test_thermal_relaxation_validation() -> None:
    invalidos = [
        {"t1": 50e-6, "t2": 120e-6, "gate_time": 1e-7},
        {"per_qubit": {"3": {"t2": 2e-4}}, "t1": 50e-6},
        {"t1": -1e-6},
        {"t1": 0},
        {"gate_time": -1e-9},
        {"per_gate": {"cx": {"t1": 1e-4}}},
        {"per_qubit": {"0": {"gate_time": 1e-7}}},
    ]
    for d in invalidos:
        try:
            NoiseModel.from_dict(d)
        except ValueError:
            continue
        raise AssertionError(f"aceito: {d}")
    sem_duracao = NoiseModel.from_dict({"t1": 1e-4, "t2": 1e-4})
    assert not sem_duracao.quantum
    assert NoiseModel.from_dict({"t1": 1e-4, "gate_time": 1e-7}).quantum
    assert NoiseModel.from_dict({"per_qubit": {"0": {"t2": 1e-4}},
                                 "per_gate": {"cx": {"gate_time": 3e-7}}}
                                ).quantum
    m = NoiseModel.from_dict({"per_qubit": {"0": {"t2": 1e-4}},
                              "per_gate": {"cx": {"gate_time": 3e-7}}})
    assert m.probs_for("h", 0) == (0.0, 0.0, 0.0)
    assert m.probs_for("cnot", 1) == (0.0, 0.0, 0.0)
    assert m.probs_for("cnot", 0)[2] > 0
    print("[ok] relaxamento termico: T2 > 2 T1, tempos invalidos e campos "
          "no lugar errado sao rejeitados; duracao so onde declarada")


# ---------------------------------------------------------------------------
# Paridade por nome de porta (inclui apelidos e portas via unitary)
# ---------------------------------------------------------------------------
def _gate_op(name: str) -> dict:
    key = gates.canonical(name)
    if key in gates.FIXED:
        _, nq = gates.FIXED[key]
        params = None
    else:
        _, nq, npar = gates.PARAMETRIC[key]
        params = [0.0] * npar
    op = {"gate": name, "targets": list(range(nq))}
    if params:
        op["params"] = params
    return op


def test_every_gate_name_gets_noise_in_both_engines() -> None:
    """Para cada nome aceito, depolarizante so naquela porta (por apelido
    no per_gate quando a porta usa o nome canonico, e vice-versa). Os
    alvos partem de |0...0>; a fracao fora de |0...0> deve bater com
    1-(1-2p/3)^k nos dois motores. Antes, rzz/rxx/id davam 0 no Aer."""
    p = 0.3
    nomes = sorted(set(gates.FIXED) | set(gates.PARAMETRIC))
    for nome in nomes:
        op = _gate_op(nome)
        k = len(op["targets"])
        # O per_gate usa o mesmo nome do circuito (apelido ou canonico):
        # os dois lados passam pela canonicalizacao.
        chave_ruido = nome
        resultados = {}
        for motor in ENGINES:
            r = run_circuit({
                "num_qubits": 3, "shots": SHOTS, "seed": 21, "backend": motor,
                "noise": {"per_gate": {chave_ruido: {"depolarizing": p}}},
                "operations": [op],
            })
            sem_ruido = run_circuit({
                "num_qubits": 3, "shots": SHOTS, "seed": 21, "backend": motor,
                "operations": [op],
            })
            resultados[motor] = (r.counts, sem_ruido.counts)
        for motor, (com, sem) in resultados.items():
            alvo = max(sem, key=sem.get)
            if sem.get(alvo, 0) == SHOTS:   # porta deterministica a partir de |0>
                esperado = 1 - (1 - 2 * p / 3) ** k
                _check(f"porta {nome}", _frac(com, lambda s: s != alvo),
                       esperado, motor)
        if len(resultados) == 2:
            a, b = resultados["numpy"][0], resultados["aer"][0]
            tvd = 0.5 * sum(abs(a.get(s, 0) - b.get(s, 0))
                            for s in set(a) | set(b)) / SHOTS
            assert tvd < 0.03, f"porta {nome}: TVD entre motores {tvd:.4f}"
    print(f"[ok] {len(nomes)} nomes de porta (com apelidos e iswap) recebem "
          "o mesmo ruido nos dois motores")


def test_per_qubit_three_qubit_gate() -> None:
    """per_qubit com ccx: antes recusado no Aer."""
    circ = {
        "num_qubits": 3, "shots": SHOTS, "seed": 31,
        "noise": {"per_qubit": {"2": {"depolarizing": 0.3}}},
        "operations": [{"gate": "x", "targets": [0]},
                       {"gate": "x", "targets": [1]},
                       {"gate": "toffoli", "targets": [0, 1, 2]}],
    }
    res = {m: run_circuit({**circ, "backend": m}).counts for m in ENGINES}
    # so a ccx toca o qubit 2: P(q2 = 0) = 2p/3 = 0,2 (string: k[0] e q2)
    for motor, counts in res.items():
        _check("per_qubit ccx", _frac(counts, lambda k: k[0] == "0"),
               2 * 0.3 / 3, motor)
    if len(res) == 2:
        a, b = res["numpy"], res["aer"]
        tvd = 0.5 * sum(abs(a.get(s, 0) - b.get(s, 0))
                        for s in set(a) | set(b)) / SHOTS
        assert tvd < 0.03, f"per_qubit + ccx: TVD {tvd:.4f}"
    print("[ok] per_qubit com porta de 3 qubits nos dois motores")


def test_unknown_gate_in_per_gate_rejected() -> None:
    try:
        run_circuit({"num_qubits": 1, "shots": 10, "backend": "numpy",
                     "noise": {"per_gate": {"cxx": {"depolarizing": 0.1}}},
                     "operations": [IDLE]})
    except ValueError:
        print("[ok] per_gate com porta desconhecida e rejeitado")
        return
    raise AssertionError("per_gate com porta desconhecida foi aceito")


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
def test_seed_independent_of_workers() -> None:
    circ = {
        "num_qubits": 3, "shots": 3000, "seed": 42, "backend": "numpy",
        "noise": {"depolarizing": 0.05, "amplitude_damping": 0.02},
        "operations": [{"gate": "h", "targets": [0]},
                       {"gate": "cx", "targets": [0, 1]},
                       {"gate": "cx", "targets": [1, 2]}],
    }
    resultados = []
    for workers in ("1", "2", "4"):
        os.environ["QSIM_TRAJECTORY_WORKERS"] = workers
        resultados.append(run_circuit(circ).counts)
    os.environ.pop("QSIM_TRAJECTORY_WORKERS", None)
    resultados.append(run_circuit({**circ, "parallel": False}).counts)
    assert all(r == resultados[0] for r in resultados), \
        "mesma seed deu contagens diferentes conforme workers/modo"
    print("[ok] mesma seed, mesmas contagens com 1, 2 e 4 workers e em "
          "modo sequencial")


# ---------------------------------------------------------------------------
# Resolucao do auto e roteamento
# ---------------------------------------------------------------------------
def test_auto_resolution() -> None:
    bell = [{"gate": "h", "targets": [0]}, {"gate": "cx", "targets": [0, 1]}]
    exato = run_circuit({"num_qubits": 2, "shots": 100, "backend": "auto",
                         "operations": bell})
    ruido = run_circuit({"num_qubits": 2, "shots": 100, "backend": "auto",
                         "noise": {"depolarizing": 0.01}, "operations": bell})
    for r in (exato, ruido):
        assert r.metadata["backend_requested"] == "auto"
    if HAS_AER:
        assert ruido.metadata["backend_resolved"] in ("aer", "aer-gpu")
    print(f"[ok] auto resolve: exato -> {exato.metadata['backend_resolved']}, "
          f"ruido -> {ruido.metadata['backend_resolved']}")


def test_scheduler_routing() -> None:
    def c(n, noise=None, ops=None):
        return {"num_qubits": n, "noise": noise or {},
                "operations": ops or [{"gate": "h", "targets": [0]}]}
    assert classify(c(5)) == "small"
    assert classify(c(5, {"depolarizing": 0.01})) == "noisy"
    assert classify(c(5, {"readout_error": 0.02})) == "small"
    assert classify(c(25, {"depolarizing": 0.01})) == "medium"
    assert classify(c(30, {"depolarizing": 0.01})) == "large"
    assert classify(c(5, ops=[{"op": "measure", "targets": [0]}])) == "noisy"
    assert classify(c(5, {"phase_damping": 0.01})) == "noisy"
    assert classify(c(5, {"t1": 1e-4, "gate_time": 1e-7})) == "noisy"
    print("[ok] roteamento: ruido em registro grande nao cai na classe noisy")


if __name__ == "__main__":
    print(f"Motores comparados: {ENGINES}\n")
    for nome, fn in list(globals().items()):
        if nome.startswith("test_") and callable(fn):
            fn()
    print("\nSuite de canais e paridade: todos os testes passaram.")
