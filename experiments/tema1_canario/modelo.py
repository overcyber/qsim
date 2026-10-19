"""Modelo fisico do canario contra injecao coerente por crosstalk.

Circuito (2 qubits):
    q1 = qubit do atacante, preparado em |1>
    q0 = canario em |+>
    k janelas ociosas; em cada uma:
        id em q0 com depolarizante p (ruido de fundo da janela)
        rzz(theta) entre q1 e q0 (acoplamento ZZ induzido pelo atacante)
    H em q0 e medicao.

Com o atacante em |1>, rzz(theta) gira o canario de theta em torno de Z.
O depolarizante e isotropico e encolhe o vetor de Bloch por (1 - 4p/3) por
janela, comutando com a rotacao. Logo

    P(1) = (1 - r cos(k theta)) / 2,   r = (1 - 4p/3)^k.

Tres fontes independentes para P(1), usadas para auditoria cruzada:
  analitico       a formula acima;
  qsim_exato      distribuicao exata por matriz densidade no Aer, usando a
                  traducao de portas e de ruido do proprio QSim
                  (services.engine.backend_aer);
  qsim_amostrado  contagens pela API publica run_circuit (o caminho que um
                  usuario do QSim percorre).
"""
from __future__ import annotations

import math
from functools import lru_cache

from services.engine.simulator import NoiseModel, run_circuit

JANELAS_PADRAO = 10


def operacoes_canario(theta: float, janelas: int) -> list[dict]:
    ops = [{"gate": "x", "targets": [1]}, {"gate": "h", "targets": [0]}]
    for _ in range(janelas):
        ops.append({"gate": "id", "targets": [0]})
        if theta:
            ops.append({"gate": "rzz", "targets": [1, 0], "params": [theta]})
    ops.append({"gate": "h", "targets": [0]})
    return ops


def circuito_canario(theta: float, janelas: int, p_fundo: float,
                     shots: int, seed: int | None,
                     backend: str = "auto") -> dict:
    circ = {"num_qubits": 2, "shots": int(shots), "backend": backend,
            "operations": operacoes_canario(theta, janelas)}
    if seed is not None:
        circ["seed"] = int(seed)
    if p_fundo > 0:
        circ["noise"] = {"per_gate": {"id": {"depolarizing": float(p_fundo)}}}
    return circ


def p1_analitico(theta: float, janelas: int, p_fundo: float) -> float:
    r = (1.0 - 4.0 * p_fundo / 3.0) ** janelas
    return (1.0 - r * math.cos(janelas * theta)) / 2.0


@lru_cache(maxsize=65536)
def _p1_exato_cache(theta: float, janelas: int, p_fundo: float) -> float:
    from qiskit_aer import AerSimulator

    from services.engine import backend_aer as ba

    ops = operacoes_canario(theta, janelas)
    native = ba.native_gates_for("density_matrix")
    qc = ba.build_qiskit_circuit(2, ops, with_clbits=False, native=native)
    qc.save_probabilities([0])
    kwargs = {"method": "density_matrix", "precision": "double"}
    if p_fundo > 0:
        noise = NoiseModel.from_dict(
            {"per_gate": {"id": {"depolarizing": p_fundo}}})
        kwargs["noise_model"] = ba.build_noise_model(noise, 2, ops,
                                                     native=native)
    res = AerSimulator(**kwargs).run(qc, shots=1).result()
    probs = res.data(0)["probabilities"]
    return float(probs[1])


def p1_qsim_exato(theta: float, janelas: int, p_fundo: float) -> float:
    """P(1) exato pela matriz densidade com a fisica do QSim."""
    return _p1_exato_cache(round(float(theta), 12), int(janelas),
                           round(float(p_fundo), 12))


def p1_qsim_amostrado(theta: float, janelas: int, p_fundo: float,
                      shots: int, seed: int,
                      backend: str = "auto") -> tuple[float, dict]:
    """Estimativa por contagens pela API publica; retorna (p, metadados)."""
    res = run_circuit(circuito_canario(theta, janelas, p_fundo, shots, seed,
                                       backend))
    uns = sum(v for k, v in res.counts.items() if k[-1] == "1")
    return uns / shots, res.metadata


class ModeloCanario:
    """Mapa (theta, p_fundo) -> P(1) usado pelo benchmark.

    fonte = "qsim_exato" (padrao) ou "analitico". As duas coincidem a 1e-10
    neste circuito (tests/test_tema1_canario.py); a fonte QSim e a que
    generaliza para canais sem forma fechada (T1/T2, crosstalk multiqubit).
    """

    def __init__(self, janelas: int = JANELAS_PADRAO,
                 fonte: str = "qsim_exato"):
        if fonte not in ("qsim_exato", "analitico"):
            raise ValueError("fonte deve ser qsim_exato ou analitico")
        self.janelas = janelas
        self.fonte = fonte

    def p1(self, theta: float, p_fundo: float) -> float:
        if self.fonte == "analitico":
            return p1_analitico(theta, self.janelas, p_fundo)
        return p1_qsim_exato(theta, self.janelas, p_fundo)
