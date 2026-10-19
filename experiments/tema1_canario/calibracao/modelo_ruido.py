"""Conversao de dados de calibracao em ruido de fundo do canario.

Um qubit ocioso por tau segundos sofre relaxamento (T1) e defasagem (T2).
A fidelidade media desse canal e

    F = (3 + exp(-tau/T1) + 2 exp(-tau/T2)) / 6

(medida sobre estados puros uniformes). O canal depolarizante do QSim
(X, Y ou Z com probabilidade p/3 cada) tem F = 1 - 2p/3. Igualando as
fidelidades medias, o depolarizante equivalente por janela e

    p = (3/2) (1 - F) = (3 - exp(-tau/T1) - 2 exp(-tau/T2)) / 4.

E uma aproximacao de primeira ordem: preserva a fidelidade media, mas
troca um canal nao unital (relaxamento) por um unital. O erro de leitura
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
