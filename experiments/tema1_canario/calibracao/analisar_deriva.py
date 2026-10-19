"""Deriva de calibracao e descasamento entre qubits por distancia no chip.

Entrada: o CSV e o mapa produzidos por coletar_ibm.py.

Passos:
  1. Converte T1/T2 de cada (snapshot, qubit) no depolarizante equivalente
     por janela ociosa do canario (modelo_ruido.p_idle). A duracao da
     janela tau e escolhida para que a mediana do chip no primeiro
     snapshot seja p_alvo (0,002, o mesmo do benchmark); como p cresce
     quase linearmente com tau nessa faixa, as razoes medidas abaixo
     praticamente nao dependem dessa escolha.
  2. Deriva: L_q(t) = log(p_q(t+1) / p_q(t)) entre snapshots consecutivos
     separados por no maximo `max_intervalo_h` horas.
  3. Modo comum: mediana de L sobre o chip em cada transicao; fracao da
     variancia de L explicada por ela.
  4. Descasamento por distancia d no mapa de acoplamento (1, 2, 3, 4+):
     para cada par (i, j), inclinacao beta_ij = cov(L_i, L_j) / var(L_i) e
     correlacao rho_ij. beta e o kappa do benchmark: a fracao da deriva do
     canario proximo que aparece na referencia. Mediana por distancia com
     IC de 95% por bootstrap sobre transicoes.
  5. Amostras empiricas de pares (p_i(t), p_i(t+1), p_j(t), p_j(t+1)) por
     distancia, usadas por benchmark_empirico.py.

Uso:
    PYTHONPATH=. python3 experiments/tema1_canario/calibracao/analisar_deriva.py \\
        --backend ibm_torino
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

from experiments.tema1_canario.calibracao.modelo_ruido import (p_idle,
                                                               tau_para_p)
from experiments.tema1_canario.manifesto import manifesto

DADOS = Path(__file__).parent / "dados"
RESULTADOS = Path(__file__).parent.parent / "resultados"
FAIXAS = ["1", "2", "3", "4+"]


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def carregar(csv_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(snapshots ordenados, T1 [S x Q], T2 [S x Q]) com NaN onde faltar."""
    por_snap: dict[str, dict[int, tuple]] = {}
    qmax = -1
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            chave = r["calibracao_utc"] or r["consulta_utc"]
            q = int(r["qubit"])
            qmax = max(qmax, q)
            t1 = float(r["t1_s"]) if r["t1_s"] else np.nan
            t2 = float(r["t2_s"]) if r["t2_s"] else np.nan
            por_snap.setdefault(chave, {})[q] = (t1, t2)
    snaps = sorted(por_snap)
    t1 = np.full((len(snaps), qmax + 1), np.nan)
    t2 = np.full_like(t1, np.nan)
    for s, chave in enumerate(snaps):
        for q, (a, b) in por_snap[chave].items():
            t1[s, q], t2[s, q] = a, b
    return snaps, t1, t2


def distancias(n: int, arestas: list[list[int]]) -> np.ndarray:
    viz = [[] for _ in range(n)]
    for a, b in arestas:
        if a < n and b < n:
            viz[a].append(b)
            viz[b].append(a)
    d = np.full((n, n), -1, dtype=int)
    for o in range(n):
        d[o, o] = 0
        fila = deque([o])
        while fila:
            u = fila.popleft()
            for v in viz[u]:
                if d[o, v] < 0:
                    d[o, v] = d[o, u] + 1
                    fila.append(v)
    return d


def _horas(a: str, b: str) -> float:
    from datetime import datetime
    try:
        ta = datetime.fromisoformat(a)
        tb = datetime.fromisoformat(b)
        return (tb - ta).total_seconds() / 3600
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Estatistica
# ---------------------------------------------------------------------------
def _cov_pareada(L: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Covariancia, variancias e contagem por par, com dados faltantes."""
    M = np.isfinite(L).astype(float)
    X = np.where(M > 0, L, 0.0)
    n = M.T @ M
    sx = X.T @ M
    sxy = X.T @ X
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (sxy - sx * sx.T / n) / (n - 1)
        sxx = (X * X).T @ M
        var_i = (sxx - sx ** 2 / n) / (n - 1)
    return cov, var_i, n


def indices_por_faixa(D: np.ndarray) -> dict:
    """Pares (i < j) de cada faixa de distancia, calculados uma vez."""
    iu, ju = np.triu_indices(D.shape[0], k=1)
    dist = D[iu, ju]
    out = {}
    for faixa in FAIXAS:
        if faixa == "4+":
            m = dist >= 4
        else:
            m = dist == int(faixa)
        out[faixa] = (iu[m], ju[m])
    return out


def metricas_por_distancia(L: np.ndarray, idx_faixas: dict,
                           min_pares_tempo: int = 8) -> dict:
    cov, var_i, n = _cov_pareada(L)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = cov / var_i                   # beta[i, j]: deriva de j sobre a de i
        rho = cov / np.sqrt(var_i * var_i.T)
    out = {}
    for faixa, (ii, jj) in idx_faixas.items():
        ok = n[ii, jj] >= min_pares_tempo
        ii, jj = ii[ok], jj[ok]
        if ii.size == 0:
            out[faixa] = {"pares": 0}
            continue
        b = np.concatenate([beta[ii, jj], beta[jj, ii]])
        out[faixa] = {"pares": int(ii.size),
                      "beta_mediano": float(np.nanmedian(b)),
                      "rho_mediano": float(np.nanmedian(rho[ii, jj]))}
    return out


def bootstrap(L: np.ndarray, idx_faixas: dict, reps: int,
              rng: np.random.Generator) -> dict:
    acum = {f: {"beta": [], "rho": []} for f in FAIXAS}
    T = L.shape[0]
    for _ in range(reps):
        idx = rng.integers(0, T, size=T)
        m = metricas_por_distancia(L[idx], idx_faixas)
        for f in FAIXAS:
            if m[f].get("pares"):
                acum[f]["beta"].append(m[f]["beta_mediano"])
                acum[f]["rho"].append(m[f]["rho_mediano"])
    ic = {}
    for f in FAIXAS:
        ic[f] = {k: [float(np.nanpercentile(v, 2.5)),
                     float(np.nanpercentile(v, 97.5))] if v else None
                 for k, v in acum[f].items()}
    return ic


# ---------------------------------------------------------------------------
# Analise
# ---------------------------------------------------------------------------
def analisar(snaps: list[str], t1: np.ndarray, t2: np.ndarray,
             arestas: list[list[int]], p_alvo: float = 0.002,
             max_intervalo_h: float = 36.0, reps_bootstrap: int = 300,
             amostras_max: int = 5000, seed: int = 20261019) -> dict:
    rng = np.random.default_rng(seed)
    S, Q = t1.shape

    # tau: mediana do chip no primeiro snapshot com dados igual a p_alvo
    s0 = next(s for s in range(S) if np.isfinite(t1[s]).sum() > Q // 2)
    t1m = float(np.nanmedian(t1[s0]))
    t2m = float(np.nanmedian(np.minimum(t2[s0], 2 * t1[s0])))
    tau = tau_para_p(p_alvo, t1m, t2m)
    P = p_idle(t1, t2, tau)

    # transicoes consecutivas dentro do intervalo maximo
    pares_t = [(s, s + 1) for s in range(S - 1)
               if _horas(snaps[s], snaps[s + 1]) <= max_intervalo_h
               or not np.isfinite(_horas(snaps[s], snaps[s + 1]))]
    if len(pares_t) < 5:
        raise ValueError("menos de 5 transicoes entre calibracoes; colete mais dias")
    a = np.array([s for s, _ in pares_t])
    b = np.array([s for _, s in pares_t])
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.log(P[b] / P[a])
    L[~np.isfinite(L)] = np.nan

    # magnitude da deriva (razao p(t+1)/p(t))
    R = np.exp(L[np.isfinite(L)])
    deriva = {f"q{int(q * 100):02d}": float(np.quantile(R, q))
              for q in (0.05, 0.10, 0.50, 0.90, 0.95)}
    deriva["frac_aumento_acima_50pct"] = float(np.mean(R > 1.5))

    # modo comum
    c = np.nanmedian(L, axis=1, keepdims=True)
    resid = L - c
    var_total = float(np.nanvar(L))
    frac_comum = 1 - float(np.nanvar(resid)) / var_total if var_total > 0 else 0.0

    D = distancias(Q, arestas)
    idx_faixas = indices_por_faixa(D)
    por_dist = metricas_por_distancia(L, idx_faixas)
    ic = bootstrap(L, idx_faixas, reps_bootstrap, rng)
    for f in FAIXAS:
        if por_dist[f].get("pares"):
            por_dist[f]["beta_ic95"] = ic[f]["beta"]
            por_dist[f]["rho_ic95"] = ic[f]["rho"]

    # amostras empiricas de pares (p_i(t), p_i(t+1), p_j(t), p_j(t+1))
    amostras = {}
    for f in FAIXAS:
        ii, jj = idx_faixas[f]
        # os dois sentidos: qualquer um dos dois qubits pode ser o proximo
        ii, jj = np.concatenate([ii, jj]), np.concatenate([jj, ii])
        if len(ii) == 0:
            amostras[f] = []
            continue
        k = rng.integers(0, len(ii), size=amostras_max * 4)
        t = rng.integers(0, len(pares_t), size=amostras_max * 4)
        quad = np.stack([P[a[t], ii[k]], P[b[t], ii[k]],
                         P[a[t], jj[k]], P[b[t], jj[k]]], axis=1)
        quad = quad[np.all(np.isfinite(quad) & (quad > 0), axis=1)]
        amostras[f] = quad[:amostras_max].round(8).tolist()

    return {
        "parametros": {"p_alvo": p_alvo, "tau_s": tau,
                       "max_intervalo_h": max_intervalo_h,
                       "reps_bootstrap": reps_bootstrap, "seed": seed},
        "snapshots": S, "qubits": Q, "transicoes": len(pares_t),
        "p_mediano_chip": float(np.nanmedian(P)),
        "p_q10_q90_chip": [float(np.nanquantile(P, 0.1)),
                           float(np.nanquantile(P, 0.9))],
        "deriva_razao_por_transicao": deriva,
        "fracao_variancia_modo_comum": frac_comum,
        "por_distancia": por_dist,
        "amostras_pares": amostras,
    }


def imprimir(res: dict) -> None:
    print(f"{res['snapshots']} snapshots, {res['qubits']} qubits, "
          f"{res['transicoes']} transicoes; tau = {res['parametros']['tau_s'] * 1e9:.0f} ns")
    print(f"p por janela: mediana {res['p_mediano_chip']:.4f}, "
          f"q10-q90 {res['p_q10_q90_chip'][0]:.4f}-{res['p_q10_q90_chip'][1]:.4f}")
    d = res["deriva_razao_por_transicao"]
    print(f"razao p(t+1)/p(t): q05 {d['q05']:.2f}, q50 {d['q50']:.2f}, "
          f"q95 {d['q95']:.2f}; aumento > 50% em {d['frac_aumento_acima_50pct']:.1%}")
    print(f"variancia da deriva explicada pelo modo comum: "
          f"{res['fracao_variancia_modo_comum']:.1%}")
    print("\n  distancia  pares   beta (kappa) [IC95]          rho [IC95]")
    for f in FAIXAS:
        m = res["por_distancia"][f]
        if not m.get("pares"):
            print(f"  {f:>9}  0")
            continue
        bi, ri = m.get("beta_ic95") or [np.nan] * 2, m.get("rho_ic95") or [np.nan] * 2
        print(f"  {f:>9}  {m['pares']:5d}   {m['beta_mediano']:+.3f} "
              f"[{bi[0]:+.3f}, {bi[1]:+.3f}]   {m['rho_mediano']:+.3f} "
              f"[{ri[0]:+.3f}, {ri[1]:+.3f}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--max-intervalo-h", type=float, default=36.0)
    args = ap.parse_args()
    csv_path = DADOS / f"{args.backend}.csv"
    mapa = json.loads((DADOS / f"{args.backend}_mapa.json").read_text())
    snaps, t1, t2 = carregar(csv_path)
    res = analisar(snaps, t1, t2, mapa["arestas"],
                   max_intervalo_h=args.max_intervalo_h)
    res = {"manifesto": manifesto({"backend": args.backend,
                                   "fonte_dados": str(csv_path.name)}), **res}
    imprimir(res)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    saida = RESULTADOS / f"deriva_{args.backend}.json"
    saida.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\nresultado em {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
