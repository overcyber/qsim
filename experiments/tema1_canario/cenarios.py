"""Cenarios do benchmark: deriva de calibracao e perfis de atacante.

Linha do tempo de cada replica: W janelas de monitoramento; as janelas
0..inicio-1 calibram a base; o ataque (se houver) comeca em tau.

Deriva de calibracao: o ruido de fundo do canario proximo cresce
linearmente de p_fundo ate p_fundo * (1 + deriva) ao longo do horizonte.
O canario de referencia recebe a fracao kappa dessa deriva (kappa = 1:
deriva comum identica; kappa < 1: qubits distantes derivam menos, o caso
que pressiona o detector diferencial).

Perfis de atacante (theta = rotacao ZZ por janela ociosa do canario):
  constante     theta fixo a partir de tau;
  rampa         theta cresce linearmente de 0 em tau ate theta no fim,
                imitando deriva para enganar referencias moveis;
  intermitente  theta em rajadas: ativo em uma fracao do periodo.

As probabilidades por janela vem do ModeloCanario (fisica do QSim); as
contagens sao Binomial(n, p) por janela, o que e exato para shots
independentes (verificado em piloto_binomial.py contra jobs do QSim).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from experiments.tema1_canario.modelo import ModeloCanario


@dataclass(frozen=True)
class Cenario:
    nome: str
    descricao: str
    deriva: float = 0.0
    kappa: float = 1.0
    ataque: str = "nenhum"          # nenhum | constante | rampa | intermitente
    theta: float = 0.0
    periodo: int = 20
    fracao_ativa: float = 0.25

    def to_dict(self) -> dict:
        return asdict(self)


def theta_por_janela(cen: Cenario, W: int, tau: int) -> np.ndarray:
    th = np.zeros(W)
    if cen.ataque == "nenhum":
        return th
    w = np.arange(W)
    ativo = w >= tau
    if cen.ataque == "constante":
        th[ativo] = cen.theta
    elif cen.ataque == "rampa":
        th[ativo] = cen.theta * (w[ativo] - tau + 1) / (W - tau)
    elif cen.ataque == "intermitente":
        fase = (w - tau) % cen.periodo
        th[ativo & (fase < cen.fracao_ativa * cen.periodo)] = cen.theta
    else:
        raise ValueError(f"ataque desconhecido: {cen.ataque}")
    return th


def probabilidades(cen: Cenario, modelo: ModeloCanario, p_fundo: float,
                   W: int, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """(p_proximo, p_referencia) por janela."""
    rampa = np.arange(W) / max(W - 1, 1)
    pf_prox = p_fundo * (1 + cen.deriva * rampa)
    pf_ref = p_fundo * (1 + cen.kappa * cen.deriva * rampa)
    th = theta_por_janela(cen, W, tau)
    p_prox = np.array([modelo.p1(t, p) for t, p in zip(th, pf_prox)])
    p_ref = np.array([modelo.p1(0.0, p) for p in pf_ref])
    return p_prox, p_ref


def amostrar(p: np.ndarray, n: int, replicas: int,
             rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(n, np.broadcast_to(p, (replicas, p.size)))
