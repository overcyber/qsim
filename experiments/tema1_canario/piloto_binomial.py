"""Reproducao versionada do piloto binomial do Tema 1.

Piloto original (18/10/2026): canario com 10 janelas ociosas, ruido de
fundo depolarizante de 0,2% por janela e injecao ZZ de theta rad por
janela; poder de um teste binomial unilateral com falso positivo de 1%,
calculado por aproximacao normal sobre probabilidades estimadas com a mesma
seed em todos os pontos.

Esta versao:
  1. calcula P(1) por tres fontes independentes (analitica, exata pela
     matriz densidade do QSim e amostrada pela API publica), com seed
     distinta por ponto para nao correlacionar os desvios;
  2. recalcula o poder pela aproximacao normal do piloto e pelo teste
     binomial exato (tamanho real <= alfa);
  3. verifica empiricamente tamanho e poder com 1000 jobs do QSim de n
     shots cada por hipotese; reporta o intervalo de Wilson de 95% e usa o
     de 99,9% como criterio de aprovacao (quatro checagens a 95% falhariam
     por acaso em cerca de 19% das execucoes);
  4. grava tudo com manifesto de proveniencia em
     resultados/piloto_binomial.json.

Uso:
    PYTHONPATH=. python3 experiments/tema1_canario/piloto_binomial.py
    PYTHONPATH=. python3 experiments/tema1_canario/piloto_binomial.py --verificar
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from experiments.tema1_canario.manifesto import manifesto
from experiments.tema1_canario.modelo import (p1_analitico, p1_qsim_amostrado,
                                              p1_qsim_exato)

SAIDA = Path(__file__).parent / "resultados" / "piloto_binomial.json"

CONFIG = {
    "janelas": 10,
    "p_fundo": 0.002,
    "thetas": [0.0, 0.02, 0.04, 0.06, 0.1],
    "shots_estimativa": 200_000,
    "alfa": 0.01,
    "n_canario": [500, 2000, 8000],
    "seed_mestre": 20261018,
    "empirico": {"theta": 0.02, "n_canario": [500, 2000], "replicas": 1000},
    "backend": "auto",
}


# ---------------------------------------------------------------------------
# Estatistica
# ---------------------------------------------------------------------------
def _log_pmf(n: int, p: float) -> np.ndarray:
    k = np.arange(n + 1)
    if p <= 0.0:
        out = np.full(n + 1, -np.inf)
        out[0] = 0.0
        return out
    if p >= 1.0:
        out = np.full(n + 1, -np.inf)
        out[n] = 0.0
        return out
    lg = np.array([math.lgamma(i + 1) for i in range(n + 1)])
    return (lg[n] - lg - lg[::-1]) + k * math.log(p) + (n - k) * math.log1p(-p)


def cauda_superior(n: int, p: float) -> np.ndarray:
    """S[c] = P(X >= c), X ~ Binomial(n, p), para c = 0..n+1."""
    pmf = np.exp(_log_pmf(n, p))
    s = np.concatenate([np.cumsum(pmf[::-1])[::-1], [0.0]])
    return np.minimum(s, 1.0)


def teste_binomial_exato(n: int, p0: float, p1: float,
                         alfa: float) -> dict:
    s0 = cauda_superior(n, p0)
    c = int(np.argmax(s0 <= alfa))          # menor c com P(X>=c | p0) <= alfa
    return {"limiar_contagem": c, "tamanho_real": float(s0[c]),
            "poder": float(cauda_superior(n, p1)[c])}


def poder_normal(p0: float, p1: float, n: int, alfa: float) -> float:
    """Aproximacao do piloto original (unilateral, normal)."""
    z_alfa = _z_quantil(1 - alfa)
    limiar = p0 + z_alfa * math.sqrt(p0 * (1 - p0) / n)
    s1 = math.sqrt(p1 * (1 - p1) / n)
    if s1 == 0:
        return float(p1 > limiar)
    return 0.5 * math.erfc((limiar - p1) / s1 / math.sqrt(2))


def _z_quantil(q: float) -> float:
    """Quantil da normal padrao por bisseccao (sem scipy)."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    f = k / n
    den = 1 + z * z / n
    centro = (f + z * z / (2 * n)) / den
    meia = z * math.sqrt(f * (1 - f) / n + z * z / (4 * n * n)) / den
    return max(0.0, centro - meia), min(1.0, centro + meia)


def _seeds(seed_mestre: int, rotulo: int, quantidade: int) -> list[int]:
    ss = np.random.SeedSequence(seed_mestre, spawn_key=(rotulo,))
    return [int(s) for s in ss.generate_state(quantidade, dtype=np.uint32)]


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------
def executar(cfg: dict = CONFIG) -> dict:
    k, pf, alfa = cfg["janelas"], cfg["p_fundo"], cfg["alfa"]
    seeds_pontos = _seeds(cfg["seed_mestre"], 0, len(cfg["thetas"]))
    motores = set()

    pontos = []
    base_exata = p1_qsim_exato(0.0, k, pf)
    base_amostrada = None
    for theta, seed in zip(cfg["thetas"], seeds_pontos):
        p_an = p1_analitico(theta, k, pf)
        p_ex = p1_qsim_exato(theta, k, pf)
        p_am, meta = p1_qsim_amostrado(theta, k, pf, cfg["shots_estimativa"],
                                       seed, cfg["backend"])
        motores.add(meta.get("backend_resolved", meta.get("backend", "?")))
        sig = math.sqrt(max(p_an * (1 - p_an), 1e-15) / cfg["shots_estimativa"])
        if theta == 0.0:
            base_amostrada = p_am
        ponto = {"theta": theta, "seed": seed, "p1_analitico": p_an,
                 "p1_qsim_exato": p_ex, "p1_qsim_amostrado": p_am,
                 "z_amostrado_vs_analitico": (p_am - p_an) / sig,
                 "poder": {}}
        if theta > 0:
            for n in cfg["n_canario"]:
                exato = teste_binomial_exato(n, base_exata, p_ex, alfa)
                ponto["poder"][str(n)] = {
                    "normal_piloto": poder_normal(base_amostrada, p_am, n, alfa),
                    "binomial_exato": exato["poder"],
                    "limiar_contagem": exato["limiar_contagem"],
                    "tamanho_real": exato["tamanho_real"],
                }
        pontos.append(ponto)

    # Verificacao empirica independente: jobs reais de n shots no QSim.
    emp_cfg = cfg["empirico"]
    theta_e = emp_cfg["theta"]
    p1_e = p1_qsim_exato(theta_e, k, pf)
    empirico = []
    for idx, n in enumerate(emp_cfg["n_canario"]):
        exato = teste_binomial_exato(n, base_exata, p1_e, alfa)
        c = exato["limiar_contagem"]
        rej = {}
        for rot, theta in (("h0", 0.0), ("h1", theta_e)):
            seeds = _seeds(cfg["seed_mestre"], 100 + 2 * idx + (rot == "h1"),
                           emp_cfg["replicas"])
            alarmes = 0
            for s in seeds:
                p_hat, _ = p1_qsim_amostrado(theta, k, pf, n, s, cfg["backend"])
                alarmes += int(round(p_hat * n)) >= c
            lo, hi = wilson(alarmes, emp_cfg["replicas"])
            lo3, hi3 = wilson(alarmes, emp_cfg["replicas"], z=3.290527)
            rej[rot] = {"alarmes": alarmes, "replicas": emp_cfg["replicas"],
                        "taxa": alarmes / emp_cfg["replicas"],
                        "ic95_wilson": [lo, hi],
                        "ic999_wilson": [lo3, hi3]}
        empirico.append({
            "n_canario": n, "theta": theta_e, "limiar_contagem": c,
            "tamanho_teorico": exato["tamanho_real"],
            "poder_teorico": exato["poder"],
            "falso_positivo_empirico": rej["h0"],
            "deteccao_empirica": rej["h1"],
            "tamanho_dentro_ic999": rej["h0"]["ic999_wilson"][0]
            <= exato["tamanho_real"] <= rej["h0"]["ic999_wilson"][1],
            "poder_dentro_ic999": rej["h1"]["ic999_wilson"][0]
            <= exato["poder"] <= rej["h1"]["ic999_wilson"][1],
        })

    return {
        "manifesto": manifesto({"motores_resolvidos": sorted(motores)}),
        "config": cfg,
        "pontos": pontos,
        "empirico": empirico,
    }


def imprimir(res: dict) -> None:
    cfg = res["config"]
    print(f"Canario: {cfg['janelas']} janelas, fundo {cfg['p_fundo']}, "
          f"alfa {cfg['alfa']}; motor(es) {res['manifesto']['motores_resolvidos']}")
    print("  theta   P1 analitico  P1 exato QSim  P1 amostrado  z")
    for p in res["pontos"]:
        print(f"  {p['theta']:5.3f}   {p['p1_analitico']:.6f}      "
              f"{p['p1_qsim_exato']:.6f}       {p['p1_qsim_amostrado']:.6f}  "
              f"{p['z_amostrado_vs_analitico']:+.2f}")
    print("\n  poder (normal do piloto / binomial exato)")
    print("  theta   " + "   ".join(f"n={n:<13}" for n in cfg["n_canario"]))
    for p in res["pontos"]:
        if not p["poder"]:
            continue
        cel = [f"{v['normal_piloto']:.3f}/{v['binomial_exato']:.3f}"
               for v in p["poder"].values()]
        print(f"  {p['theta']:5.3f}   " + "      ".join(f"{c:<11}" for c in cel))
    print("\n  verificacao empirica (jobs do QSim)")
    for e in res["empirico"]:
        h0, h1 = e["falso_positivo_empirico"], e["deteccao_empirica"]
        print(f"  n={e['n_canario']:<5} limiar c={e['limiar_contagem']:<4} "
              f"FP teorico {e['tamanho_teorico']:.4f} empirico {h0['taxa']:.4f} "
              f"IC [{h0['ic95_wilson'][0]:.4f}, {h0['ic95_wilson'][1]:.4f}] | "
              f"poder teorico {e['poder_teorico']:.3f} empirico {h1['taxa']:.3f} "
              f"IC [{h1['ic95_wilson'][0]:.3f}, {h1['ic95_wilson'][1]:.3f}]")


def verificar(ref: dict, novo: dict) -> list[str]:
    """Compara uma nova execucao com a referencia gravada."""
    falhas = []
    mesmo_ambiente = (
        ref["manifesto"]["motores_resolvidos"] == novo["manifesto"]["motores_resolvidos"]
        and ref["manifesto"]["versoes"]["qiskit_aer"]
        == novo["manifesto"]["versoes"]["qiskit_aer"])
    for a, b in zip(ref["pontos"], novo["pontos"]):
        if abs(a["p1_qsim_exato"] - b["p1_qsim_exato"]) > 1e-12:
            falhas.append(f"theta={a['theta']}: P1 exato mudou")
        if mesmo_ambiente and a["p1_qsim_amostrado"] != b["p1_qsim_amostrado"]:
            falhas.append(f"theta={a['theta']}: amostra mudou com mesma seed e motor")
        if abs(b["z_amostrado_vs_analitico"]) > 5:
            falhas.append(f"theta={a['theta']}: amostra a mais de 5 sigma")
    for e in novo["empirico"]:
        if not (e["tamanho_dentro_ic999"] and e["poder_dentro_ic999"]):
            falhas.append(f"n={e['n_canario']}: teoria fora do IC empirico "
                          "de 99,9%")
    return falhas


def main() -> int:
    novo = executar()
    imprimir(novo)
    if "--verificar" in sys.argv:
        if not SAIDA.exists():
            print("\nsem referencia gravada; rode sem --verificar primeiro")
            return 2
        falhas = verificar(json.loads(SAIDA.read_text()), novo)
        print("\nverificacao contra a referencia:",
              "OK" if not falhas else "FALHOU")
        for f in falhas:
            print("  -", f)
        return 1 if falhas else 0
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(novo, indent=2, ensure_ascii=False))
    print(f"\nreferencia gravada em {SAIDA.relative_to(Path.cwd()) if SAIDA.is_relative_to(Path.cwd()) else SAIDA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
