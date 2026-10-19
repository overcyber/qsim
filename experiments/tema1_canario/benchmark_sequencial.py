"""Benchmark sequencial de detectores de injecao por crosstalk.

Protocolo:
  1. Fisica: P(1) por janela vem do ModeloCanario (matriz densidade com a
     traducao de ruido do QSim; a fonte analitica coincide a 1e-10), com
     erro de leitura de 1% aplicado ao bit medido em todos os cenarios.
  2. Calibracao: para cada detector, limiar h que da probabilidade de
     falso alarme `fap` no horizonte sob o cenario estacionario sem ataque,
     estimado num conjunto de replicas de calibracao com seed propria.
  3. Avaliacao: cada cenario usa replicas novas, com seeds independentes
     das de calibracao (o limiar nunca ve os dados em que e avaliado).
     O canario multicomprimento (detector proposto) usa o mesmo total de
     n shots por janela, divididos entre tres comprimentos (alocacao que
     minimiza a variancia no ponto nominal), com circuitos mais longos.
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
    PYTHONPATH=. python3 experiments/tema1_canario/benchmark_sequencial.py --janelas 30

--janelas N troca o comprimento do canario dos detectores de base (padrao
10) e grava em benchmark_sequencial_kN.json, sem tocar na referencia.
Serve de controle: o canario multicomprimento usa circuitos de ate 60
janelas, entao a linha de base tambem e avaliada com canario mais longo.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from experiments.tema1_canario.cenarios import (Cenario, amostrar,
                                                probabilidades,
                                                probabilidades_multi)
from experiments.tema1_canario.detectores import (DETECTORES,
                                                  alocacao_multicomprimento,
                                                  calibrar_limiar,
                                                  primeiro_alarme,
                                                  theta_projeto)
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
    "comprimentos_multi": [0, 30, 60],
    "leitura_nominal": 0.01,
    "janela_variancia": 20,
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
    Cenario("S8", "erro de leitura do canario proximo de 1% a 1,5%, "
            "referencia fixa em 1%, sem ataque", deriva_leitura=0.5,
            kappa_leitura=0.0),
    Cenario("S9", "deriva de 100% no ruido e na leitura do canario proximo, "
            "referencia sem deriva, sem ataque", deriva=1.0, kappa=0.0,
            deriva_leitura=1.0, kappa_leitura=0.0),
    Cenario("S10", "atacante adaptativo: theta=0.01 constante no canario "
            "proximo e na referencia", ataque="constante", theta=0.01,
            ataque_ref=True),
    Cenario("S11", "deriva de frequencia do canario proximo: fase extra de "
            "0 a 0,01 rad por janela, sem ataque", deriva_fase=0.01),
]


def _rng(cfg: dict, rotulo: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(cfg["seed_mestre"], spawn_key=(rotulo,)))


def _escores(c: np.ndarray, f: np.ndarray, multi: list[np.ndarray],
             cfg: dict, aux: dict) -> dict:
    args = {"n": cfg["n_shots"], "inicio": cfg["inicio"],
            "delta": cfg["delta"], "janela_ref": cfg["janela_ref"],
            "defasagem": cfg["defasagem"], "f": f, "multi": multi,
            "n_multi": aux["n_multi"], "comprimentos": cfg["comprimentos_multi"],
            "theta_proj": aux["theta_proj"],
            "janela_var": cfg["janela_variancia"]}
    return {nome: fn(c, **args) for nome, fn in DETECTORES.items()}


def auxiliares(cfg: dict) -> dict:
    """Alocacao de shots do canario multicomprimento e rotacao de projeto."""
    return {
        "n_multi": alocacao_multicomprimento(
            cfg["comprimentos_multi"], cfg["p_fundo"], cfg["leitura_nominal"],
            cfg["n_shots"]),
        "theta_proj": theta_projeto(cfg["p_fundo"], cfg["janelas_ociosas"],
                                    cfg["delta"]),
    }


def _amostrar_multi(pm: np.ndarray, n_multi: list[int], replicas: int,
                    rng: np.random.Generator) -> list[np.ndarray]:
    return [amostrar(p, nm, replicas, rng) for p, nm in zip(pm, n_multi)]


def executar(cfg: dict = CONFIG, cenarios: list[Cenario] = CENARIOS) -> dict:
    modelo = ModeloCanario(cfg["janelas_ociosas"], cfg["fonte_fisica"])
    W, tau, ini, n = cfg["W"], cfg["tau"], cfg["inicio"], cfg["n_shots"]
    aux = auxiliares(cfg)
    comp = cfg["comprimentos_multi"]

    # Calibracao sob H0 estacionario, replicas exclusivas. As contagens dos
    # canarios de comprimento unico sao sorteadas antes das do canario
    # multicomprimento, entao acrescentar este nao muda as demais.
    p_prox0, p_ref0 = probabilidades(cenarios[0], modelo, cfg["p_fundo"],
                                     W, tau)
    pm0 = probabilidades_multi(cenarios[0], modelo, cfg["p_fundo"], W, tau,
                               comp)
    rng = _rng(cfg, 1000)
    c0 = amostrar(p_prox0, n, cfg["replicas_calibracao"], rng)
    f0 = amostrar(p_ref0, n, cfg["replicas_calibracao"], rng)
    m0 = _amostrar_multi(pm0, aux["n_multi"], cfg["replicas_calibracao"], rng)
    limiares = {nome: calibrar_limiar(e, ini, cfg["fap"])
                for nome, e in _escores(c0, f0, m0, cfg, aux).items()}

    resultados = []
    contexto = {}
    for i, cen in enumerate(cenarios):
        p_prox, p_ref = probabilidades(cen, modelo, cfg["p_fundo"], W, tau)
        pm = probabilidades_multi(cen, modelo, cfg["p_fundo"], W, tau, comp)
        contexto[cen.nome] = {
            "descricao": cen.descricao,
            "parametros": cen.to_dict(),
            "p1_proximo_inicio": float(p_prox[0]),
            "p1_proximo_fim": float(p_prox[-1]),
            "p1_referencia_fim": float(p_ref[-1]),
            "p1_multi_fim": [float(x) for x in pm[:, -1]],
        }
        rng = _rng(cfg, 2000 + i)
        c = amostrar(p_prox, n, cfg["replicas_teste"], rng)
        f = amostrar(p_ref, n, cfg["replicas_teste"], rng)
        mm = _amostrar_multi(pm, aux["n_multi"], cfg["replicas_teste"], rng)
        for nome, esc in _escores(c, f, mm, cfg, aux).items():
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

    return {"manifesto": manifesto(), "config": cfg, "auxiliares": aux,
            "cenarios": contexto, "limiares": limiares,
            "resultados": resultados}


def imprimir(res: dict) -> None:
    cfg = res["config"]
    print(f"n={cfg['n_shots']} shots/janela, W={cfg['W']}, tau={cfg['tau']}, "
          f"falso alarme calibrado {cfg['fap']:.0%} no horizonte, "
          f"{cfg['replicas_teste']} replicas por cenario")
    for nome, cen in res["cenarios"].items():
        print(f"  {nome}: {cen['descricao']} (P1 proximo "
              f"{cen['p1_proximo_inicio']:.4f} -> {cen['p1_proximo_fim']:.4f})")
    aux = res.get("auxiliares", {})
    if aux:
        print(f"  canario multicomprimento: comprimentos "
              f"{cfg['comprimentos_multi']}, shots {aux['n_multi']}, "
              f"theta de projeto {aux['theta_proj']:.4f} rad")
    print("\n  cen  detector                     FA      FA pre  detec   "
          "atraso med/p90")
    for r in res["resultados"]:
        if "falso_alarme" in r:
            print(f"  {r['cenario']:<4} {r['detector']:<27}  "
                  f"{r['falso_alarme']:.3f}"
                  f"   (pos-tau {r['alarme_pos_tau']:.3f})")
        else:
            med = "-" if r["atraso_mediano"] is None else f"{r['atraso_mediano']:.0f}"
            p90 = "-" if r["atraso_p90"] is None else f"{r['atraso_p90']:.0f}"
            print(f"  {r['cenario']:<4} {r['detector']:<27}  {'':5}   "
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
    chaves_ref = [(r["cenario"], r["detector"]) for r in ref["resultados"]]
    chaves_novo = [(r["cenario"], r["detector"]) for r in novo["resultados"]]
    if chaves_ref != chaves_novo:
        return ["cenarios ou detectores diferem da referencia; regrave-a"]
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
    if "--janelas" in sys.argv:
        k = int(sys.argv[sys.argv.index("--janelas") + 1])
        res = executar({**CONFIG, "janelas_ociosas": k})
        imprimir(res)
        saida = DIR / f"benchmark_sequencial_k{k}.json"
        DIR.mkdir(parents=True, exist_ok=True)
        saida.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\nresultado em {saida} (a referencia nao foi alterada)")
        return 0
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
