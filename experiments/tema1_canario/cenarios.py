"""Cenarios do benchmark: deriva de calibracao e perfis de atacante.

Linha do tempo de cada replica: W janelas de monitoramento; as janelas
0..inicio-1 calibram a base; o ataque (se houver) comeca em tau.

Deriva de calibracao: o ruido de fundo do canario proximo cresce
linearmente de p_fundo ate p_fundo * (1 + deriva) ao longo do horizonte.
O canario de referencia recebe a fracao kappa dessa deriva (kappa = 1:
deriva comum identica; kappa < 1: qubits distantes derivam menos, o caso
que pressiona o detector diferencial).

Erro de leitura: 1% em todos os canarios por padrao (a faixa tipica de
processadores supercondutores e de 1% a 3%); sem ele, o canario de
comprimento 0 seria deterministico e a calibracao irrealista.

Deriva de leitura: o erro de leitura do canario proximo cresce de
`leitura` ate leitura * (1 + deriva_leitura); a referencia recebe a
fracao kappa_leitura dessa deriva (0: erros de leitura independentes).

Deriva de frequencia: o canario proximo acumula uma fase extra por janela
que cresce de 0 ate `deriva_fase` rad (dessintonia do qubit ou ZZ de um
vizinho legitimo). Fisicamente e um deslocamento coerente de fase, como o
ataque, e serve para medir esse confundidor.

Perfis de atacante (theta = rotacao ZZ por janela ociosa do canario):
  constante     theta fixo a partir de tau;
  rampa         theta cresce linearmente de 0 em tau ate theta no fim,
                imitando deriva para enganar referencias moveis;
  intermitente  theta em rajadas: ativo em uma fracao do periodo.
Com ataque_ref = True, o atacante tambem acopla a referencia com o mesmo
theta (atacante adaptativo contra o detector diferencial).

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
    leitura: float = 0.01
    deriva_leitura: float = 0.0
    kappa_leitura: float = 0.0
    ataque_ref: bool = False
    deriva_fase: float = 0.0

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


def trajetorias(cen: Cenario, p_fundo: float, W: int, tau: int) -> dict:
    """Ruido de fundo, erro de leitura e fase por janela de cada canario."""
    rampa = np.arange(W) / max(W - 1, 1)
    th = theta_por_janela(cen, W, tau)
    return {
        "p_prox": p_fundo * (1 + cen.deriva * rampa),
        "p_ref": p_fundo * (1 + cen.kappa * cen.deriva * rampa),
        "e_prox": cen.leitura * (1 + cen.deriva_leitura * rampa),
        "e_ref": cen.leitura * (1 + cen.kappa_leitura * cen.deriva_leitura
                                * rampa),
        "fase_prox": th + cen.deriva_fase * rampa,
        "fase_ref": th if cen.ataque_ref else np.zeros(W),
    }


def probabilidades(cen: Cenario, modelo: ModeloCanario, p_fundo: float,
                   W: int, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """(p_proximo, p_referencia) por janela, no comprimento do modelo."""
    t = trajetorias(cen, p_fundo, W, tau)
    p_prox = np.array([modelo.p1(f, p, leitura=e) for f, p, e in
                       zip(t["fase_prox"], t["p_prox"], t["e_prox"])])
    p_ref = np.array([modelo.p1(f, p, leitura=e) for f, p, e in
                      zip(t["fase_ref"], t["p_ref"], t["e_ref"])])
    return p_prox, p_ref


def probabilidades_multi(cen: Cenario, modelo: ModeloCanario, p_fundo: float,
                         W: int, tau: int,
                         comprimentos: list[int]) -> np.ndarray:
    """P(1) do canario proximo em cada comprimento: matriz (M, W)."""
    t = trajetorias(cen, p_fundo, W, tau)
    return np.array([[modelo.p1(f, p, janelas=m, leitura=e) for f, p, e in
                      zip(t["fase_prox"], t["p_prox"], t["e_prox"])]
                     for m in comprimentos])


def amostrar(p: np.ndarray, n: int, replicas: int,
             rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(n, np.broadcast_to(p, (replicas, p.size)))
