"""Benchmark dos detectores com deriva e bases tiradas da calibracao real.

Cada replica sorteia um par real de qubits a distancia d no chip e uma
transicao real entre calibracoes consecutivas (amostras de
analisar_deriva.py): o canario proximo parte do ruido de base do qubit i e
termina no valor da calibracao seguinte; a referencia faz o mesmo com o
qubit j. Nada de deriva ou kappa fixados por hipotese, e as bases dos dois
canarios sao diferentes, como no hardware.

Cenarios por distancia d:
  E0  sem deriva (bases reais heterogeneas), sem ataque: calibra limiares
  E1  deriva real, sem ataque
  E3  deriva real + ataque constante theta = 0,01
  E7  deriva real + rampa lenta ate theta = 0,015

Erro de leitura fixo em `leitura_nominal` (1%) nos dois canarios, como no
benchmark sequencial; a deriva vem so de T1/T2. O canario multicomprimento
(detector proposto) usa o mesmo qubit proximo e o mesmo total de shots.

Fisica: P(1) pela formula fechada do canario, vetorizada. Antes de rodar,
ela e conferida contra a matriz densidade do QSim em duas frentes, e
diferenca acima de 1e-10 aborta:
  - com o depolarizante, em pontos sorteados da faixa de p usada;
  - com o canal de relaxamento termico (T1, T2, duracao da janela), em
    pontos de T1 e T2 sorteados, contra a conversao p_coerencia usada por
    analisar_deriva.py. Isso valida que o ruido de fundo vindo da
    calibracao real e o do canal fisico, nao uma aproximacao.

Uso:
    PYTHONPATH=. python3 experiments/tema1_canario/calibracao/benchmark_empirico.py \\
        --backend ibm_torino
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiments.tema1_canario import benchmark_sequencial as bench
from experiments.tema1_canario.calibracao.modelo_ruido import p_coerencia
from experiments.tema1_canario.cenarios import Cenario, theta_por_janela
from experiments.tema1_canario.detectores import (DETECTORES, calibrar_limiar,
                                                  primeiro_alarme)
from experiments.tema1_canario.manifesto import manifesto
from experiments.tema1_canario.modelo import (p1_qsim_exato,
                                              p1_qsim_relaxacao_exato)

RESULTADOS = Path(__file__).parent.parent / "resultados"

ATAQUES = {
    "E1": Cenario("E1", "deriva real, sem ataque"),
    "E3": Cenario("E3", "deriva real + ataque constante theta=0.01",
                  ataque="constante", theta=0.01),
    "E7": Cenario("E7", "deriva real + rampa lenta ate theta=0.015",
                  ataque="rampa", theta=0.015),
}


def p1_vetor(theta, p, janelas: int, leitura: float = 0.0) -> np.ndarray:
    r = (1.0 - 4.0 * np.asarray(p) / 3.0) ** janelas
    p1 = (1.0 - r * np.cos(janelas * np.asarray(theta))) / 2.0
    return leitura + (1.0 - 2.0 * leitura) * p1


def conferir_fisica(amostras: np.ndarray, janelas: int,
                    rng: np.random.Generator, pontos: int = 60) -> float:
    ps = amostras[rng.integers(0, len(amostras), pontos), rng.integers(0, 4, pontos)]
    thetas = rng.uniform(0, 0.05, pontos)
    return float(max(abs(p1_vetor(t, p, janelas) - p1_qsim_exato(t, janelas, p))
                     for t, p in zip(thetas, ps)))


def conferir_relaxacao(tau: float, janelas: int, rng: np.random.Generator,
                       pontos: int = 30) -> float:
    """Formula com p_coerencia contra o canal T1/T2 do QSim."""
    t1 = rng.uniform(40e-6, 400e-6, pontos)
    t2 = t1 * rng.uniform(0.1, 2.0, pontos)
    thetas = rng.uniform(0, 0.05, pontos)
    return float(max(
        abs(p1_vetor(th, p_coerencia(a, b, tau), janelas)
            - p1_qsim_relaxacao_exato(th, janelas, a, b, tau))
        for th, a, b in zip(thetas, t1, t2)))


def _probabilidades(quad: np.ndarray, theta_w: np.ndarray, W: int,
                    janelas: int, deriva: bool, cfg: dict
                    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    frac = np.arange(W) / max(W - 1, 1) if deriva else np.zeros(W)
    pn = quad[:, [0]] + (quad[:, [1]] - quad[:, [0]]) * frac
    pf = quad[:, [2]] + (quad[:, [3]] - quad[:, [2]]) * frac
    e = cfg["leitura_nominal"]
    multi = [p1_vetor(theta_w[None, :], pn, m, e)
             for m in cfg["comprimentos_multi"]]
    return (p1_vetor(theta_w[None, :], pn, janelas, e),
            p1_vetor(0.0, pf, janelas, e), multi)


def _amostrar(rng, n, pn, pf, multi, n_multi):
    c = rng.binomial(n, pn)
    f = rng.binomial(n, np.broadcast_to(pf, pn.shape))
    mm = [rng.binomial(nm, np.broadcast_to(pm, pn.shape))
          for pm, nm in zip(multi, n_multi)]
    return c, f, mm


def _escores(c, f, mm, cfg, aux):
    args = {"n": cfg["n_shots"], "inicio": cfg["inicio"],
            "delta": cfg["delta"], "janela_ref": cfg["janela_ref"],
            "defasagem": cfg["defasagem"], "f": f, "multi": mm,
            "n_multi": aux["n_multi"], "comprimentos": cfg["comprimentos_multi"],
            "theta_proj": aux["theta_proj"],
            "janela_var": cfg["janela_variancia"]}
    return {nome: fn(c, **args) for nome, fn in DETECTORES.items()}


def executar(deriva_json: dict, cfg: dict = bench.CONFIG) -> dict:
    W, tau, ini, n = cfg["W"], cfg["tau"], cfg["inicio"], cfg["n_shots"]
    k = cfg["janelas_ociosas"]
    aux = bench.auxiliares(cfg)
    R = cfg["replicas_teste"]
    seq = np.random.SeedSequence(cfg["seed_mestre"], spawn_key=(3000,))
    rng_fis, *_ = [np.random.default_rng(s) for s in seq.spawn(1)]

    resultados, conferencias = [], {}
    params = deriva_json.get("parametros", {})
    if params.get("modelo_ruido", "coerencia") == "coerencia" and \
            params.get("tau_s"):
        conferencias["relaxacao_T1_T2"] = conferir_relaxacao(
            float(params["tau_s"]), k, rng_fis)
        if conferencias["relaxacao_T1_T2"] > 1e-10:
            raise RuntimeError("formula com p_coerencia diverge do canal T1/T2 "
                               f"do QSim: {conferencias['relaxacao_T1_T2']:.2e}")
    for fi, faixa in enumerate(deriva_json["amostras_pares"]):
        amostras = np.asarray(deriva_json["amostras_pares"][faixa], dtype=float)
        if len(amostras) < 100:
            continue
        conferencias[faixa] = conferir_fisica(amostras, k, rng_fis)
        if conferencias[faixa] > 1e-10:
            raise RuntimeError(f"formula fechada diverge do QSim em {faixa}: "
                               f"{conferencias[faixa]:.2e}")
        rng = np.random.default_rng(np.random.SeedSequence(
            cfg["seed_mestre"], spawn_key=(3100 + fi,)))

        def sortear(r):
            return amostras[rng.integers(0, len(amostras), r)]

        # calibracao: bases reais, sem deriva, sem ataque, replicas proprias
        qc = sortear(cfg["replicas_calibracao"])
        pc, pfc, mc = _probabilidades(qc, np.zeros(W), W, k, False, cfg)
        c0, f0, m0 = _amostrar(rng, n, pc, pfc, mc, aux["n_multi"])
        limiares = {d: calibrar_limiar(e, ini, cfg["fap"])
                    for d, e in _escores(c0, f0, m0, cfg, aux).items()}

        for nome_cen, cen in ATAQUES.items():
            quad = sortear(R)
            th = theta_por_janela(cen, W, tau)
            pn, pf, pm = _probabilidades(quad, th, W, k, True, cfg)
            c, f, mm = _amostrar(rng, n, pn, pf, pm, aux["n_multi"])
            for det, esc in _escores(c, f, mm, cfg, aux).items():
                alarme = primeiro_alarme(esc, limiares[det], ini)
                linha = {"distancia": faixa, "cenario": nome_cen,
                         "detector": det, "limiar": limiares[det]}
                if cen.ataque == "nenhum":
                    linha["falso_alarme"] = float(np.mean(alarme >= 0))
                    linha["alarme_pos_tau"] = float(np.mean(alarme >= tau))
                else:
                    det_mask = alarme >= tau
                    atraso = alarme[det_mask] - tau
                    linha.update({
                        "falso_alarme_pre": float(np.mean((alarme >= 0)
                                                          & (alarme < tau))),
                        "deteccao": float(np.mean(det_mask)),
                        "atraso_mediano": float(np.median(atraso))
                        if atraso.size else None})
                resultados.append(linha)

    return {"manifesto": manifesto({"fonte_deriva": deriva_json.get(
                "manifesto", {}).get("backend", "desconhecido")}),
            "config": cfg, "conferencia_fisica_max_dif": conferencias,
            "resultados": resultados}


def imprimir(res: dict) -> None:
    print("conferencia formula x QSim (maior diferenca por faixa):",
          {k: f"{v:.1e}" for k, v in res["conferencia_fisica_max_dif"].items()})
    dets = list(DETECTORES)
    curto = {"shewhart": "shew", "cusum": "cusum", "cusum_movel": "movel",
             "cusum_diferencial": "dif", "cusum_diferencial_calibrado": "dif_cal",
             "cusum_multicomprimento": "multi"}
    print("\n  d    cen  " + "  ".join(f"{curto[d]:>13}" for d in dets))
    linhas = {}
    for r in res["resultados"]:
        linhas.setdefault((r["distancia"], r["cenario"]), {})[r["detector"]] = r
    for (faixa, cen), por_det in linhas.items():
        cel = []
        for d in dets:
            r = por_det[d]
            if "falso_alarme" in r:
                cel.append(f"FA {r['falso_alarme']:.3f}")
            else:
                cel.append(f"det {r['deteccao']:.3f}")
        print(f"  {faixa:<4} {cen}   " + "  ".join(f"{x:>13}" for x in cel))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    args = ap.parse_args()
    entrada = RESULTADOS / f"deriva_{args.backend}.json"
    res = executar(json.loads(entrada.read_text()))
    imprimir(res)
    saida = RESULTADOS / f"benchmark_empirico_{args.backend}.json"
    saida.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\nresultado em {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
