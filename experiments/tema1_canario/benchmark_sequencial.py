"""Benchmark sequencial de detectores de injecao por crosstalk.

Protocolo:
  1. Fisica: P(1) por janela vem do ModeloCanario (matriz densidade com a
     traducao de ruido do QSim; a fonte analitica coincide a 1e-10).
  2. Calibracao: para cada detector, limiar h que da probabilidade de
     falso alarme `fap` no horizonte sob o cenario estacionario sem ataque,
     estimado num conjunto de replicas de calibracao com seed propria.
  3. Avaliacao: cada cenario usa replicas novas, com seeds independentes
     das de calibracao (o limiar nunca ve os dados em que e avaliado).
  4. Metricas por detector e cenario:
       sem ataque  falso_alarme = P(algum alarme em [inicio, W))
                   alarme_pos_tau = P(primeiro alarme em [tau, W)): base
                   para ler a deteccao dos cenarios com ataque sob a mesma
                   deriva (deteccao atribuivel ao ataque = deteccao do
                   cenario com ataque menos alarme_pos_tau do cenario sem
                   ataque com a mesma deriva)
       com ataque  falso_alarme_pre = P(alarme em [inicio, tau))
                   deteccao = P(primeiro alarme em [tau, W))
                   atraso mediano e p90 (janelas apos tau), entre detectados
  5. Tudo e gravado com manifesto de proveniencia.

Uso:
    PYTHONPATH=. python3 experiments/tema1_canario/benchmark_sequencial.py
    PYTHONPATH=. python3 experiments/tema1_canario/benchmark_sequencial.py --verificar
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from experiments.tema1_canario.cenarios import (Cenario, amostrar,
                                                probabilidades)
from experiments.tema1_canario.detectores import (DETECTORES, calibrar_limiar,
                                                  primeiro_alarme)
from experiments.tema1_canario.manifesto import manifesto
from experiments.tema1_canario.modelo import ModeloCanario

DIR = Path(__file__).parent / "resultados"
SAIDA_JSON = DIR / "benchmark_sequencial.json"
SAIDA_CSV = DIR / "benchmark_sequencial.csv"

CONFIG = {
    "janelas_ociosas": 10,
    "p_fundo": 0.002,
    "n_shots": 500,
    "W": 300,
    "tau": 100,
    "inicio": 50,
    "delta": 0.25,
    "janela_ref": 40,
    "defasagem": 10,
    "fap": 0.05,
    "replicas_calibracao": 4000,
    "replicas_teste": 4000,
    "seed_mestre": 20261018,
    "fonte_fisica": "qsim_exato",
}

CENARIOS = [
    Cenario("S0", "estacionario, sem ataque"),
    Cenario("S1", "deriva de 50% comum aos canarios, sem ataque",
            deriva=0.5, kappa=1.0),
    Cenario("S2", "deriva de 50% com descasamento (referencia deriva 50% "
            "menos), sem ataque", deriva=0.5, kappa=0.5),
    Cenario("S3", "ataque constante theta=0.01, estacionario",
            ataque="constante", theta=0.01),
    Cenario("S4", "ataque constante theta=0.01 com deriva comum de 50%",
            deriva=0.5, ataque="constante", theta=0.01),
    Cenario("S5", "ataque em rampa ate theta=0.03 com deriva comum de 50%",
            deriva=0.5, ataque="rampa", theta=0.03),
    Cenario("S6", "ataque intermitente theta=0.03, 25% de um periodo de 20 "
            "janelas, estacionario", ataque="intermitente", theta=0.03),
    Cenario("S7", "ataque em rampa lenta ate theta=0.015, estacionario "
            "(atacante que tenta treinar referencias moveis)",
            ataque="rampa", theta=0.015),
]


def _rng(cfg: dict, rotulo: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(cfg["seed_mestre"], spawn_key=(rotulo,)))


def _escores(c: np.ndarray, f: np.ndarray, cfg: dict) -> dict:
    args = {"n": cfg["n_shots"], "inicio": cfg["inicio"],
            "delta": cfg["delta"], "janela_ref": cfg["janela_ref"],
            "defasagem": cfg["defasagem"], "f": f}
    return {nome: fn(c, **args) for nome, fn in DETECTORES.items()}


def executar(cfg: dict = CONFIG, cenarios: list[Cenario] = CENARIOS) -> dict:
    modelo = ModeloCanario(cfg["janelas_ociosas"], cfg["fonte_fisica"])
    W, tau, ini, n = cfg["W"], cfg["tau"], cfg["inicio"], cfg["n_shots"]

    # Calibracao sob H0 estacionario, replicas exclusivas.
    p_prox0, p_ref0 = probabilidades(cenarios[0], modelo, cfg["p_fundo"],
                                     W, tau)
    rng = _rng(cfg, 1000)
    c0 = amostrar(p_prox0, n, cfg["replicas_calibracao"], rng)
    f0 = amostrar(p_ref0, n, cfg["replicas_calibracao"], rng)
    limiares = {nome: calibrar_limiar(e, ini, cfg["fap"])
                for nome, e in _escores(c0, f0, cfg).items()}

    resultados = []
    for i, cen in enumerate(cenarios):
        p_prox, p_ref = probabilidades(cen, modelo, cfg["p_fundo"], W, tau)
        rng = _rng(cfg, 2000 + i)
        c = amostrar(p_prox, n, cfg["replicas_teste"], rng)
        f = amostrar(p_ref, n, cfg["replicas_teste"], rng)
        for nome, esc in _escores(c, f, cfg).items():
            alarme = primeiro_alarme(esc, limiares[nome], ini)
            linha = {"cenario": cen.nome, "detector": nome,
                     "limiar": limiares[nome]}
            if cen.ataque == "nenhum":
                linha["falso_alarme"] = float(np.mean(alarme >= 0))
                # Alarmes a partir de tau sem ataque: base de comparacao para
                # a deteccao dos cenarios com ataque sob a mesma deriva.
                linha["alarme_pos_tau"] = float(np.mean(alarme >= tau))
            else:
                pre = (alarme >= 0) & (alarme < tau)
                det = alarme >= tau
                atraso = alarme[det] - tau
                linha.update({
                    "falso_alarme_pre": float(np.mean(pre)),
                    "deteccao": float(np.mean(det)),
                    "atraso_mediano": float(np.median(atraso))
                    if atraso.size else None,
                    "atraso_p90": float(np.percentile(atraso, 90))
                    if atraso.size else None,
                })
            resultados.append(linha)

    contexto = {}
    for cen in cenarios:
        p_prox, p_ref = probabilidades(cen, modelo, cfg["p_fundo"], W, tau)
        contexto[cen.nome] = {
            "descricao": cen.descricao,
            "parametros": cen.to_dict(),
            "p1_proximo_inicio": float(p_prox[0]),
            "p1_proximo_fim": float(p_prox[-1]),
            "p1_referencia_fim": float(p_ref[-1]),
        }

    return {"manifesto": manifesto(), "config": cfg, "cenarios": contexto,
            "limiares": limiares, "resultados": resultados}


def imprimir(res: dict) -> None:
    cfg = res["config"]
    print(f"n={cfg['n_shots']} shots/janela, W={cfg['W']}, tau={cfg['tau']}, "
          f"falso alarme calibrado {cfg['fap']:.0%} no horizonte, "
          f"{cfg['replicas_teste']} replicas por cenario")
    for nome, cen in res["cenarios"].items():
        print(f"  {nome}: {cen['descricao']} (P1 proximo "
              f"{cen['p1_proximo_inicio']:.4f} -> {cen['p1_proximo_fim']:.4f})")
    print("\n  cen  detector            FA      FA pre  detec   atraso med/p90")
    for r in res["resultados"]:
        if "falso_alarme" in r:
            print(f"  {r['cenario']}   {r['detector']:<18}  {r['falso_alarme']:.3f}"
                  f"   (pos-tau {r['alarme_pos_tau']:.3f})")
        else:
            med = "-" if r["atraso_mediano"] is None else f"{r['atraso_mediano']:.0f}"
            p90 = "-" if r["atraso_p90"] is None else f"{r['atraso_p90']:.0f}"
            print(f"  {r['cenario']}   {r['detector']:<18}  {'':5}   "
                  f"{r['falso_alarme_pre']:.3f}   {r['deteccao']:.3f}   "
                  f"{med}/{p90}")


def gravar(res: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    SAIDA_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    campos = ["cenario", "detector", "limiar", "falso_alarme",
              "alarme_pos_tau", "falso_alarme_pre", "deteccao",
              "atraso_mediano", "atraso_p90"]
    with SAIDA_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for r in res["resultados"]:
            w.writerow({k: r.get(k) for k in campos})


def verificar(ref: dict, novo: dict) -> list[str]:
    falhas = []
    if ref["manifesto"]["versoes"]["numpy"] != novo["manifesto"]["versoes"]["numpy"]:
        print("aviso: numpy difere da referencia "
              f"({ref['manifesto']['versoes']['numpy']} contra "
              f"{novo['manifesto']['versoes']['numpy']}); o gerador pode "
              "mudar entre versoes, comparacao exata ignorada")
        return []
    for a, b in zip(ref["resultados"], novo["resultados"]):
        for k in ("limiar", "falso_alarme", "alarme_pos_tau",
                  "falso_alarme_pre", "deteccao", "atraso_mediano",
                  "atraso_p90"):
            va, vb = a.get(k), b.get(k)
            if va is None and vb is None:
                continue
            if va is None or vb is None or abs(va - vb) > 1e-12:
                falhas.append(f"{a['cenario']}/{a['detector']}/{k}: "
                              f"{va} != {vb}")
    return falhas


def main() -> int:
    novo = executar()
    imprimir(novo)
    if "--verificar" in sys.argv:
        if not SAIDA_JSON.exists():
            print("\nsem referencia gravada; rode sem --verificar primeiro")
            return 2
        falhas = verificar(json.loads(SAIDA_JSON.read_text()), novo)
        print("\nverificacao contra a referencia:",
              "OK" if not falhas else "FALHOU")
        for f in falhas:
            print("  -", f)
        return 1 if falhas else 0
    gravar(novo)
    print(f"\nreferencia gravada em {SAIDA_JSON} e {SAIDA_CSV.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
