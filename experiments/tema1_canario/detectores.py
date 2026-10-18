"""Detectores sequenciais sobre contagens de canarios.

Todas as funcoes operam sobre matrizes (R, W): R replicas de Monte Carlo,
W janelas de monitoramento. c[r, w] = numero de shots com resultado 1 no
canario proximo (exposto ao atacante) na janela w, de n shots; f[r, w] = o
mesmo para o canario de referencia distante (sujeito a mesma deriva, sem
acoplamento com o atacante).

Cada detector devolve um escore por janela; o alarme e o primeiro w >=
inicio com escore > h. O limiar h e calibrado por Monte Carlo para uma
probabilidade de falso alarme fixa no horizonte (calibrar_limiar).

  shewhart          z da contagem contra a base estimada na fase de
                    calibracao (janelas 0..inicio-1); sem memoria.
  cusum             CUSUM de Page com a razao de verossimilhanca binomial
                    entre p0 (base estimada) e p1 = p0 (1 + delta).
  cusum_movel       o mesmo CUSUM com p0 reestimado numa janela movel de L
                    janelas, defasada de g. Absorve deriva, mas tambem
                    absorve um ataque que cresce devagar.
  cusum_diferencial CUSUM condicional: dado t = c + f, c ~ Bin(t, q) com
                    q = p_prox / (p_prox + p_ref). Sem ataque q = 1/2 em
                    qualquer nivel de deriva comum; com ataque q > 1/2.
                    Nao usa base estimada.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _p0_calibracao(c: np.ndarray, n: int, inicio: int) -> np.ndarray:
    p0 = c[:, :inicio].sum(axis=1) / (inicio * n)
    return np.clip(p0, _EPS, 1 - _EPS)[:, None]


def _llr_binomial(c: np.ndarray, n: int, p0: np.ndarray,
                  p1: np.ndarray) -> np.ndarray:
    p0 = np.clip(p0, _EPS, 1 - _EPS)
    p1 = np.clip(p1, _EPS, 1 - _EPS)
    return c * np.log(p1 / p0) + (n - c) * np.log((1 - p1) / (1 - p0))


def _cusum(llr: np.ndarray, inicio: int) -> np.ndarray:
    s = np.full(llr.shape, -np.inf)
    acc = np.zeros(llr.shape[0])
    for w in range(inicio, llr.shape[1]):
        acc = np.maximum(0.0, acc + llr[:, w])
        s[:, w] = acc
    return s


def shewhart(c: np.ndarray, n: int, inicio: int, **_) -> np.ndarray:
    p0 = _p0_calibracao(c, n, inicio)
    z = (c - n * p0) / np.sqrt(n * p0 * (1 - p0))
    z[:, :inicio] = -np.inf
    return z


def cusum(c: np.ndarray, n: int, inicio: int, delta: float, **_) -> np.ndarray:
    p0 = _p0_calibracao(c, n, inicio)
    return _cusum(_llr_binomial(c, n, p0, p0 * (1 + delta)), inicio)


def cusum_movel(c: np.ndarray, n: int, inicio: int, delta: float,
                janela_ref: int, defasagem: int, **_) -> np.ndarray:
    if janela_ref + defasagem > inicio:
        raise ValueError("janela_ref + defasagem deve caber antes de inicio")
    soma = np.concatenate([np.zeros((c.shape[0], 1)), np.cumsum(c, axis=1)],
                          axis=1)
    p_ref = np.full(c.shape, np.nan)
    for w in range(inicio, c.shape[1]):
        fim = w - defasagem
        p_ref[:, w] = (soma[:, fim] - soma[:, fim - janela_ref]) / (
            janela_ref * n)
    llr = np.zeros(c.shape)
    llr[:, inicio:] = _llr_binomial(c[:, inicio:], n, p_ref[:, inicio:],
                                    p_ref[:, inicio:] * (1 + delta))
    return _cusum(llr, inicio)


def cusum_diferencial(c: np.ndarray, n: int, inicio: int, delta: float,
                      f: np.ndarray, **_) -> np.ndarray:
    q1 = (1 + delta) / (2 + delta)
    llr = c * np.log(2 * q1) + f * np.log(2 * (1 - q1))
    return _cusum(llr, inicio)


DETECTORES = {
    "shewhart": shewhart,
    "cusum": cusum,
    "cusum_movel": cusum_movel,
    "cusum_diferencial": cusum_diferencial,
}


def calibrar_limiar(escore: np.ndarray, inicio: int, fap: float) -> float:
    """Limiar com P(algum alarme em [inicio, W)) = fap sob H0."""
    maximos = escore[:, inicio:].max(axis=1)
    return float(np.quantile(maximos, 1 - fap, method="higher"))


def primeiro_alarme(escore: np.ndarray, h: float, inicio: int) -> np.ndarray:
    """Indice da primeira janela >= inicio com escore > h; -1 se nenhuma.

    A comparacao e estrita porque o limiar vem de um quantil 'higher' dos
    maximos de calibracao: escores discretos iguais ao limiar nao disparam.
    """
    marca = escore[:, inicio:] > h
    tem = marca.any(axis=1)
    idx = np.where(tem, marca.argmax(axis=1) + inicio, -1)
    return idx
