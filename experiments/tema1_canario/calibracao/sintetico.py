"""Historico de calibracao sintetico com verdade conhecida.

Gera um chip em grade (linhas x colunas, vizinhanca de 4) e uma serie de
snapshots diarios em que a variacao de log p de cada qubit entre dias e

    L_t ~ Normal(0, Sigma),  Sigma = a^2 11' + b^2 K K' + c^2 I

com K = I + alfa * A / grau (suavizacao espacial sobre o mapa). O primeiro
termo e o modo comum, o segundo e a parte que vizinhos compartilham e o
terceiro e idiossincratico (defeitos locais). Como p e quase proporcional a
1/T1 com T2 = T1, a serie de T1 e log T1(t+1) = log T1(t) - L_t.

A inclinacao teorica beta_ij = Sigma_ij / Sigma_ii permite conferir que
analisar_deriva.py recupera o kappa por distancia.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from experiments.tema1_canario.calibracao.analisar_deriva import (
    FAIXAS, distancias, indices_por_faixa)


def grade(linhas: int, colunas: int) -> list[list[int]]:
    arestas = []
    for r in range(linhas):
        for c in range(colunas):
            q = r * colunas + c
            if c + 1 < colunas:
                arestas.append([q, q + 1])
            if r + 1 < linhas:
                arestas.append([q, q + colunas])
    return arestas


def covariancia(Q: int, arestas: list[list[int]], a: float, b: float,
                c: float, alfa: float) -> np.ndarray:
    A = np.zeros((Q, Q))
    for i, j in arestas:
        A[i, j] = A[j, i] = 1
    grau = A.sum(axis=1, keepdims=True)
    K = np.eye(Q) + alfa * A / np.maximum(grau, 1)
    return a * a * np.ones((Q, Q)) + b * b * K @ K.T + c * c * np.eye(Q)


def beta_teorico_por_faixa(sigma: np.ndarray,
                           arestas: list[list[int]]) -> dict:
    Q = sigma.shape[0]
    beta = sigma / np.diag(sigma)[:, None]
    idx = indices_por_faixa(distancias(Q, arestas))
    out = {}
    for f in FAIXAS:
        ii, jj = idx[f]
        if ii.size:
            out[f] = float(np.median(np.concatenate([beta[ii, jj],
                                                     beta[jj, ii]])))
    return out


def gerar(destino: Path, nome: str = "sintetico", linhas: int = 6,
          colunas: int = 8, dias: int = 300, a: float = 0.05,
          b: float = 0.12, c: float = 0.15, alfa: float = 1.5,
          falta: float = 0.02, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    Q = linhas * colunas
    arestas = grade(linhas, colunas)
    sigma = covariancia(Q, arestas, a, b, c, alfa)
    Lc = np.linalg.cholesky(sigma)
    log_t1 = np.log(rng.uniform(120e-6, 300e-6, size=Q))
    inicio = dt.datetime(2026, 1, 1, 8, tzinfo=dt.timezone.utc)
    destino.mkdir(parents=True, exist_ok=True)
    with (destino / f"{nome}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["backend", "consulta_utc", "calibracao_utc", "qubit",
                    "t1_s", "t2_s", "readout_error", "sx_error"])
        for d in range(dias):
            quando = (inicio + dt.timedelta(days=d)).isoformat()
            t1 = np.exp(log_t1)
            for q in range(Q):
                if rng.random() < falta:
                    w.writerow([nome, quando, quando, q, "", "", "", ""])
                else:
                    w.writerow([nome, quando, quando, q, f"{t1[q]:.9e}",
                                f"{t1[q]:.9e}", "0.01", "0.0003"])
            log_t1 = log_t1 - Lc @ rng.standard_normal(Q)
    (destino / f"{nome}_mapa.json").write_text(json.dumps(
        {"backend": nome, "num_qubits": Q, "arestas": arestas}))
    return {"sigma": sigma, "arestas": arestas,
            "beta_teorico": beta_teorico_por_faixa(sigma, arestas),
            "frac_comum_teorica": float(a * a / np.mean(np.diag(sigma)))}
