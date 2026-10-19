"""Conversao de dados de calibracao em ruido de fundo do canario.

Um qubit ocioso por tau segundos sofre relaxamento (T1) e defasagem (T2).

Modelo usado (p_coerencia, exato para o canario). O canario prepara |+>,
acumula a fase do acoplamento ZZ e mede na base X. O resultado depende so
da coerencia rho_01, que sob relaxamento termico decai por exp(-tau/T2)
por janela, qualquer que seja T1 (T1 so altera populacoes, que a medida em
X nao ve). O depolarizante com o mesmo decaimento de coerencia e

    p = (3/4) (1 - exp(-tau/T2)),

e com ele a formula fechada do canario fica exata. Isso e conferido contra
o canal de relaxamento termico do QSim (t1, t2, gate_time) em
benchmark_empirico.py e em tests/test_tema1_calibracao.py. Consequencia
para o artigo: a deriva que importa para este canario e a de T2.

Modelo alternativo (p_idle, aproximacao generica). A fidelidade media do
canal de relaxamento e

    F = (3 + exp(-tau/T1) + 2 exp(-tau/T2)) / 6

(medida sobre estados puros uniformes). O canal depolarizante do QSim
(X, Y ou Z com probabilidade p/3 cada) tem F = 1 - 2p/3. Igualando as
fidelidades medias, o depolarizante equivalente por janela e

    p = (3/2) (1 - F) = (3 - exp(-tau/T1) - 2 exp(-tau/T2)) / 4.

E uma aproximacao de primeira ordem: preserva a fidelidade media, mas
troca um canal nao unital (relaxamento) por um unital, e para o canario
mistura T1 numa grandeza que so depende de T2. O erro de leitura
nao entra aqui; ele afeta as contagens de forma estacionaria e e absorvido
pela base calibrada dos detectores.

T2 fisico satisfaz T2 <= 2 T1; valores reportados acima disso (ruido de
estimativa da calibracao) sao limitados a 2 T1.
"""
from __future__ import annotations

import math

import numpy as np


def p_idle(t1_s, t2_s, tau_s):
    """Depolarizante equivalente por janela ociosa de tau segundos.

    Aceita escalares ou arrays; T1 e T2 em segundos. Valores ausentes,
    nulos ou negativos resultam em NaN.
    """
    t1 = np.asarray(t1_s, dtype=float)
    t2 = np.asarray(t2_s, dtype=float)
    validos = (t1 > 0) & (t2 > 0) & np.isfinite(t1) & np.isfinite(t2)
    t1s = np.where(validos, t1, 1.0)
    t2s = np.where(validos, np.minimum(t2, 2 * t1), 1.0)
    p = (3 - np.exp(-tau_s / t1s) - 2 * np.exp(-tau_s / t2s)) / 4
    p = np.where(validos, p, np.nan)
    return float(p) if p.ndim == 0 else p


def p_coerencia(t1_s, t2_s, tau_s):
    """Depolarizante com o mesmo decaimento de coerencia por janela que o
    relaxamento termico: (3/4) (1 - exp(-tau/T2)), T2 limitado a 2 T1.

    Aceita escalares ou arrays; valores ausentes, nulos ou negativos dao
    NaN.
    """
    t1 = np.asarray(t1_s, dtype=float)
    t2 = np.asarray(t2_s, dtype=float)
    validos = (t1 > 0) & (t2 > 0) & np.isfinite(t1) & np.isfinite(t2)
    t2s = np.where(validos, np.minimum(t2, 2 * np.where(validos, t1, 1.0)),
                   1.0)
    p = np.where(validos, -0.75 * np.expm1(-tau_s / t2s), np.nan)
    return float(p) if p.ndim == 0 else p


def tau_para_p_coerencia(p_alvo: float, t1_s: float, t2_s: float) -> float:
    """Duracao da janela com p_coerencia = p_alvo (forma fechada)."""
    if not 0 < p_alvo < 0.75:
        raise ValueError("p_alvo deve estar em (0, 0,75)")
    return -min(t2_s, 2 * t1_s) * np.log1p(-4.0 * p_alvo / 3.0)


def tau_para_p(p_alvo: float, t1_s: float, t2_s: float) -> float:
    """Duracao da janela que produz p_alvo para um qubit (bisseccao)."""
    lo, hi = 0.0, 10 * max(t1_s, t2_s)
    if p_idle(t1_s, t2_s, hi) < p_alvo:
        raise ValueError("p_alvo inalcancavel com estes T1 e T2")
    for _ in range(200):
        mid = (lo + hi) / 2
        if p_idle(t1_s, t2_s, mid) < p_alvo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fidelidade_media(t1_s: float, t2_s: float, tau_s: float) -> float:
    t2 = min(t2_s, 2 * t1_s)
    return (3 + math.exp(-tau_s / t1_s) + 2 * math.exp(-tau_s / t2)) / 6
